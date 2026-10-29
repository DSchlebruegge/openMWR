"""Command selection; computation and recipe handling live in the Python API."""

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from shlex import quote


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='openMWR', description='Build datasets and run retrieval tasks in an openMWR working project.')
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create-site', help='Create site settings, dataset, retrieval and evaluation recipes.')
    create.add_argument('site', metavar='SITE', help='New site name under sites/.')
    create.add_argument('-p', '--project-dir', type=Path, default=Path('.'), help='Project containing data/ and sites/ (default: current directory).')
    migrate = commands.add_parser('migrate', help='Upgrade old site settings and saved retrieval checkpoints.')
    migrate.add_argument('-p', '--project-dir', type=Path, default=Path('.'), help='Project containing data/ and sites/ (default: current directory).')
    migrate.add_argument('-n', '--dry-run', action='store_true', help='Check all destinations and show changes without writing files.')
    data = commands.add_parser('data', help='Create or update named datasets.')
    data.add_argument('-p', '--project-dir', type=Path, default=Path('.'), help='Project containing data/ and sites/ (default: current directory).')
    data.add_argument('site', metavar='SITE', help='Site name under sites/.')
    data.add_argument('dataset', metavar='DATASET', nargs='+', help='Recipe name(s) under sites/<site>/datasets/.')
    data.add_argument('-w', '--with-dependencies', action='store_true', help='Update prerequisites first; shared inputs run once.')
    data.add_argument('-n', '--dry-run', action='store_true', help='Show processing order without writing files or contacting sources.')
    data.add_argument('-r', '--rebuild', action='store_true', help='Regenerate only selected outputs; prerequisites receive ordinary updates.')
    predict = commands.add_parser('predict', help='Apply saved retrievals to local measurements and save daily NetCDF files.')
    predict.add_argument('-p', '--project-dir', type=Path, default=Path('.'), help='Project containing data/ and sites/ (default: current directory).')
    predict.add_argument('site', metavar='SITE', help='Site name under sites/.')
    predict.add_argument('retrieval', metavar='RETRIEVAL', nargs='*', help='Saved retrieval name(s); defaults to [prediction].retrievals in site.toml.')
    predict.add_argument('-D', '--device', default='auto', help='PyTorch device: auto selects CUDA, then MPS, then CPU (default: auto).')
    dates = predict.add_mutually_exclusive_group(required=True)
    dates.add_argument('-d', '--date', type=date.fromisoformat, help='One UTC day, YYYY-MM-DD.')
    dates.add_argument('-s', '--start-date', type=date.fromisoformat, help='First UTC day of an inclusive range, YYYY-MM-DD.')
    predict.add_argument('-e', '--end-date', type=date.fromisoformat, help='Last UTC day; required with --start-date.')
    predict.add_argument('-o', '--output-dir', type=Path, help='Output root, relative to project-dir or absolute; replaces the selected daily files.')
    train = commands.add_parser('train', help='Train from a named recipe, optimizing when it defines a parameter search.')
    train.add_argument('-p', '--project-dir', type=Path, default=Path('.'), help='Project containing data/ and sites/ (default: current directory).')
    train.add_argument('site', metavar='SITE', help='Site name under sites/.')
    train.add_argument('retrieval', metavar='RETRIEVAL', help='Directory name under sites/<site>/retrieval/.')
    train.add_argument('-D', '--device', default='auto', help='PyTorch device: auto selects CUDA, then MPS, then CPU (default: auto).')
    train.add_argument('-t', '--n-trials', type=int, help='Override the trial count when the recipe defines a parameter search.')
    mode = train.add_mutually_exclusive_group()
    mode.add_argument('-f', '--fresh', dest='run_mode', action='store_const', const='fresh', help="Replace only this retrieval's model and optimization study.")
    mode.add_argument('-c', '--continue', dest='run_mode', action='store_const', const='continue', help="Add trials to this retrieval's existing optimization study.")
    train.add_argument('--data-source', choices=('radiosonde', 'era5'), help='Reference data for optimization.')
    train.add_argument('--dataset', help='Override the named analysis dataset for optimization.')
    train.add_argument('-s', '--start-date', help='First observation day for optimization, YYYY-MM-DD. Fitting/test periods come from the split recipe.')
    train.add_argument('-e', '--end-date', help='Last observation day for optimization, YYYY-MM-DD (inclusive).')
    train.add_argument('--days-of-month', type=int, nargs=2, metavar=('FIRST', 'LAST'), help='Inclusive observation day-of-month range (1 31 selects all days).')
    evaluate = commands.add_parser('evaluate', help='Compare saved retrievals using a named evaluation recipe.')
    evaluate.add_argument('site', metavar='SITE')
    evaluate.add_argument('evaluation', metavar='EVALUATION', help='Recipe name under sites/<site>/evaluations/.')
    evaluate.add_argument('-p', '--project-dir', type=Path, default=Path('.'))
    evaluate.add_argument('-r', '--retrieval', nargs='+', metavar='RETRIEVAL', help='Replace the recipe retrieval list for this run.')
    evaluate.add_argument('-D', '--device', default='auto', help='PyTorch device (default: auto).')
    evaluate.add_argument('-o', '--output-dir', type=Path, help='New directory for report.html and metrics.nc.')
    for name in ('data', 'train', 'evaluate', 'predict'):
        commands.choices[name].add_argument('-b', '--background', action='store_true',
            help='Run detached from the terminal and append console output to the task log.')
    args = parser.parse_args(argv)
    if args.command == 'data' and args.background and args.dry_run:
        parser.error('--background cannot be combined with --dry-run.')
    if args.command == 'train':
        selection = {key: getattr(args, key) for key in ('start_date', 'end_date') if getattr(args, key) is not None}
        if args.days_of_month is not None:
            selection['days_of_month'] = dict(zip(('first', 'last'), args.days_of_month))
    if args.command == 'predict':
        if args.date is not None and args.end_date is not None:
            parser.error('--end-date requires --start-date.')
        if args.start_date is not None and args.end_date is None:
            parser.error('--start-date requires --end-date.')
        if args.start_date is not None and args.end_date < args.start_date:
            parser.error('--end-date must be on or after --start-date.')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    try:
        from openMWR.background import LOG_ENV, capture_console, launch

        background_log = os.environ.pop(LOG_ENV, None)
        if getattr(args, 'background', False) and background_log is None:
            launch(args, argv)
            return
        if background_log is not None:
            capture_console(background_log)
            logging.getLogger(__name__).info('Starting %s in background (PID %s).', args.command, os.getpid())
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
        elif args.command == 'migrate':
            from openMWR.migrate import migrate_project
            sites = migrate_project(args.project_dir, dry_run=args.dry_run)
            if sites:
                print('Original JSON files, datasets, scripts and optimization databases stay in place.')
                print('Original checkpoints are kept beside model.pth as model.pth.bak or numbered model.pth.bak.N files.')
                print('Check paths in sites/<site>/site.toml. Relative paths start at the project root.')
                print('Choose a saved retrieval explicitly for prediction. No retrieval.toml is required:')
                print('  openMWR predict SITE RETRIEVAL --date YYYY-MM-DD --output-dir data/migration_prediction')
                print('For Python scripts, processed datasets and future training, follow the documentation page "Upgrading from 0.1".')
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
            from openMWR.evaluation import EvaluationRecipe, evaluate, save_evaluation
            if args.output_dir is not None and args.output_dir.exists():
                raise FileExistsError(f'Evaluation output already exists: {args.output_dir}. Choose a new directory.')
            recipe = EvaluationRecipe.load(args.project_dir, args.site, args.evaluation)
            result = evaluate(recipe, retrievals=args.retrieval, device=args.device)
            print(save_evaluation(result, args.output_dir))
        else:
            from openMWR.train import TrainingWorkflow
            workflow = TrainingWorkflow.from_recipe(args.project_dir, args.site, args.retrieval, args.device)
            if workflow.requires_optimization:
                workflow.optimize(n_trials=args.n_trials, data_source=args.data_source, dataset=args.dataset, run_mode=args.run_mode, **selection)
            else:
                if args.n_trials is not None or args.data_source is not None or args.dataset is not None or selection:
                    raise ValueError('--n-trials, --data-source, --dataset and observation-selection flags require input-noise optimization or search ranges in the recipe.')
                logging.getLogger(__name__).info('Training one model: %s.', args.retrieval)
                workflow.train(run_mode=args.run_mode)
        if background_log is not None:
            logging.getLogger(__name__).info('Finished %s command.', args.command)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(1, f'{args.command.capitalize()} failed: {exc}\n')


if __name__ == '__main__':
    main()
