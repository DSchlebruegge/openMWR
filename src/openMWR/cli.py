"""Command selection; computation and recipe handling live in the Python API."""

import argparse
import logging
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m openMWR', description='Run individual retrieval tasks in an openMWR working project.')
    commands = parser.add_subparsers(dest='command', required=True)
    for command, help_text in (
        ('train', 'Train from a named recipe, optimizing when it defines a parameter search.'),
        ('evaluate', 'Evaluate a saved retrieval and write a portable HTML report.'),
    ):
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument('--project-dir', type=Path, default=Path('.'), help='Project containing data/ and sites/ (default: current directory).')
        sub.add_argument('--site', required=True)
        sub.add_argument('--retrieval', required=True, help='Directory name under sites/<site>/retrieval/.')
        sub.add_argument('--device', default='cpu', help='PyTorch device, e.g. cpu or cuda (default: cpu).')
        if command == 'train':
            sub.add_argument('--n-trials', type=int, help='Override the trial count when the recipe defines a parameter search.')
            mode = sub.add_mutually_exclusive_group()
            mode.add_argument('--fresh', dest='run_mode', action='store_const', const='fresh', help="Replace only this retrieval's model and optimization study.")
            mode.add_argument('--continue', dest='run_mode', action='store_const', const='continue', help='Add trials to this retrieval\'s existing optimization study.')
        sub.add_argument('--data-source', choices=('radiosonde', 'era5'), help='Reference data for optimization or evaluation.')
        if command == 'evaluate':
            filters = sub.add_mutually_exclusive_group()
            filters.add_argument('--days-gt-20', dest='days_gt_20', action='store_true', default=None, help='Use observation days after the 20th.')
            filters.add_argument('--all-days', dest='days_gt_20', action='store_false', help='Use observations from all days.')
            sub.add_argument('--station-id', help='Radiosonde station ID; required when the analysis dataset has multiple stations.')
            sub.add_argument('--output', type=Path, help='New HTML file (default: dated report in the retrieval evaluation/ directory).')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    try:
        if args.command == 'evaluate':
            from openMWR.evaluation import evaluate_retrieval, write_report
            options = {key: getattr(args, key) for key in ('data_source', 'days_gt_20', 'station_id')}
            if args.output is not None and args.output.exists():
                raise FileExistsError(f'Report already exists: {args.output}. Choose a new filename.')
            report = evaluate_retrieval(args.project_dir, args.site, args.retrieval, device=args.device, **options)
            print(write_report(report, args.output))
        else:
            from openMWR.train import TrainingWorkflow
            workflow = TrainingWorkflow.from_recipe(args.project_dir, args.site, args.retrieval, args.device)
            if workflow.requires_optimization:
                workflow.optimize(n_trials=args.n_trials, data_source=args.data_source, run_mode=args.run_mode)
            else:
                if args.n_trials is not None or args.data_source is not None:
                    raise ValueError('--n-trials and --data-source require input-noise optimization or search ranges in the recipe.')
                logging.getLogger(__name__).info('Training one model: %s.', args.retrieval)
                workflow.train(run_mode=args.run_mode)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(1, f'{args.command.capitalize()} failed: {exc}\n')


if __name__ == '__main__':
    main()
