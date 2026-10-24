"""Command selection; computation and recipe handling live in the Python API."""

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path
from shlex import quote


def main(argv=None):
    parser = argparse.ArgumentParser(prog='openMWR', description='Build datasets and run retrieval tasks in an openMWR working project.')
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create-site', help='Create site settings, radiosonde dataset recipes and an example retrieval.')
    create.add_argument('site', metavar='SITE', help='New site name under sites/.')
    create.add_argument('--project-dir', type=Path, default=Path('.'), help='Project containing data/ and sites/ (default: current directory).')
    data = commands.add_parser('data', help='Create or update named datasets.')
    data.add_argument('--project-dir', type=Path, default=Path('.'), help='Project containing data/ and sites/ (default: current directory).')
    data.add_argument('site', metavar='SITE', help='Site name under sites/.')
    data.add_argument('dataset', metavar='DATASET', nargs='+', help='Recipe name(s) under sites/<site>/datasets/.')
    data.add_argument('--with-dependencies', action='store_true', help='Update prerequisites first; shared inputs run once.')
    data.add_argument('--dry-run', action='store_true', help='Show processing order without writing files or contacting sources.')
    data.add_argument('--rebuild', action='store_true', help='Regenerate only selected outputs; prerequisites receive ordinary updates.')
    predict = commands.add_parser('predict', help='Apply saved retrievals to local measurements and save daily NetCDF files.')
    predict.add_argument('--project-dir', type=Path, default=Path('.'), help='Project containing data/ and sites/ (default: current directory).')
    predict.add_argument('site', metavar='SITE', help='Site name under sites/.')
    predict.add_argument('retrieval', metavar='RETRIEVAL', nargs='*', help='Saved retrieval name(s); defaults to [prediction].retrievals in site.toml.')
    predict.add_argument('--device', default='cpu', help='PyTorch device (default: cpu).')
    dates = predict.add_mutually_exclusive_group(required=True)
    dates.add_argument('--date', type=date.fromisoformat, help='One UTC day, YYYY-MM-DD.')
    dates.add_argument('--start-date', type=date.fromisoformat, help='First UTC day of an inclusive range, YYYY-MM-DD.')
    predict.add_argument('--end-date', type=date.fromisoformat, help='Last UTC day; required with --start-date.')
    predict.add_argument('--output-dir', type=Path, help='Output root, relative to project-dir or absolute; replaces the selected daily files.')
    for command, help_text in (
        ('train', 'Train from a named recipe, optimizing when it defines a parameter search.'),
        ('evaluate', 'Evaluate a saved retrieval and write a portable HTML report.'),
    ):
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument('--project-dir', type=Path, default=Path('.'), help='Project containing data/ and sites/ (default: current directory).')
        sub.add_argument('site', metavar='SITE', help='Site name under sites/.')
        sub.add_argument('retrieval', metavar='RETRIEVAL', help='Directory name under sites/<site>/retrieval/.')
        sub.add_argument('--device', default='cpu', help='PyTorch device, e.g. cpu or cuda (default: cpu).')
        if command == 'train':
            sub.add_argument('--n-trials', type=int, help='Override the trial count when the recipe defines a parameter search.')
            mode = sub.add_mutually_exclusive_group()
            mode.add_argument('--fresh', dest='run_mode', action='store_const', const='fresh', help="Replace only this retrieval's model and optimization study.")
            mode.add_argument('--continue', dest='run_mode', action='store_const', const='continue', help='Add trials to this retrieval\'s existing optimization study.')
        sub.add_argument('--data-source', choices=('radiosonde', 'era5'), help='Reference data for optimization or evaluation.')
        sub.add_argument('--dataset', help='Override the named analysis dataset for optimization or evaluation.')
        if command == 'evaluate':
            sub.add_argument('--days-of-month', type=int, nargs=2, metavar=('FIRST', 'LAST'), help='Inclusive observation day-of-month range (1 31 selects all days).')
            for kind in ('observation', 'test'):
                sub.add_argument(f'--{kind}-height-range', nargs='+', metavar='HEIGHT', help='Profile height bounds in metres (MIN MAX), or model for the complete model grid.')
            sub.add_argument('--station-id', help='Radiosonde station ID; required when the analysis dataset has multiple stations.')
            sub.add_argument('--output', type=Path, help='New HTML file (default: dated report in the retrieval evaluation/ directory).')
    args = parser.parse_args(argv)
    if args.command == 'evaluate':
        for key in ('observation_height_range', 'test_height_range'):
            value = getattr(args, key)
            if value is not None:
                try:
                    value = 'model' if value == ['model'] else [float(bound) for bound in value]
                except ValueError:
                    parser.error(f'--{key.replace("_", "-")} requires MIN MAX or model.')
                setattr(args, key, value)
    if args.command == 'predict':
        if args.date is not None and args.end_date is not None:
            parser.error('--end-date requires --start-date.')
        if args.start_date is not None and args.end_date is None:
            parser.error('--start-date requires --end-date.')
        if args.start_date is not None and args.end_date < args.start_date:
            parser.error('--end-date must be on or after --start-date.')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    try:
        if args.command == 'create-site':
            from openMWR.site import create_site
            directory = create_site(args.site, args.project_dir, with_examples=True)
            print(f'Created site: {directory}')
            for path in sorted(directory.rglob('*.toml')):
                print(f'  {path.relative_to(directory)}')
            print(f'Shared grids: {directory.parent.parent / "grids.toml"}')
            print(f'\nEdit {directory / "site.toml"} to fill in your instrument and location settings.')
            print(f'Edit {directory / "datasets/radiosondes.toml"} to set radiosonde station IDs.')
            print('Radiative transfer also requires a working libRadtran installation.')
            print('Then check the dataset chain and build the training data:')
            print(f'  cd {quote(str(directory.parent.parent))}')
            print(f'  openMWR data {quote(args.site)} training --with-dependencies --dry-run')
            print(f'  openMWR data {quote(args.site)} training --with-dependencies')
            print(f'  openMWR train {quote(args.site)} NN_example')
        elif args.command == 'data':
            from openMWR.datasets import DatasetWorkflow
            DatasetWorkflow(args.project_dir, args.site).build(args.dataset,
                with_dependencies=args.with_dependencies, rebuild=args.rebuild, dry_run=args.dry_run)
        elif args.command == 'predict':
            from openMWR.predict import PredictionWorkflow
            workflow = PredictionWorkflow(args.project_dir, args.site, args.retrieval or None,
                args.device, output_dir=args.output_dir)
            first = args.date or args.start_date
            last = args.date or args.end_date
            failed = False
            for offset in range((last - first).days + 1):
                day = first + timedelta(days=offset)
                result = workflow.predict(day)
                for path in workflow.save(result):
                    print(path)
                failed |= bool(result.errors)
            if failed:
                parser.exit(1, 'Prediction incomplete; see the reported dates/groups. Successful output was saved.\n')
        elif args.command == 'evaluate':
            from openMWR.evaluation import evaluate_retrieval, write_report
            options = {key: getattr(args, key) for key in ('data_source', 'station_id', 'dataset', 'observation_height_range', 'test_height_range')}
            if args.days_of_month is not None:
                options['days_of_month'] = dict(zip(('first', 'last'), args.days_of_month))
            if args.output is not None and args.output.exists():
                raise FileExistsError(f'Report already exists: {args.output}. Choose a new filename.')
            report = evaluate_retrieval(args.project_dir, args.site, args.retrieval, device=args.device, **options)
            print(write_report(report, args.output))
        else:
            from openMWR.train import TrainingWorkflow
            workflow = TrainingWorkflow.from_recipe(args.project_dir, args.site, args.retrieval, args.device)
            if workflow.requires_optimization:
                workflow.optimize(n_trials=args.n_trials, data_source=args.data_source, dataset=args.dataset, run_mode=args.run_mode)
            else:
                if args.n_trials is not None or args.data_source is not None or args.dataset is not None:
                    raise ValueError('--n-trials, --data-source and --dataset require input-noise optimization or search ranges in the recipe.')
                logging.getLogger(__name__).info('Training one model: %s.', args.retrieval)
                workflow.train(run_mode=args.run_mode)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(1, f'{args.command.capitalize()} failed: {exc}\n')


if __name__ == '__main__':
    main()
