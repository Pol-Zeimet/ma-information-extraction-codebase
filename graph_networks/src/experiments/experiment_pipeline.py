from datetime import datetime
from typing import List
import mlflow
import sys

from src.experiments.experiment_class.experiment_class import BaseExperiment


class ExperimentPipeline:
    def run(self, experiments: List[BaseExperiment]):
        for i, experiment in enumerate(experiments):
            try:
                print(
                    f"{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}  Experiment {i + 1}/{len(experiments)}: {experiment.name}")
                experiment.setup()
                experiment.run()

            except:
                print(f'An error occurred while executing experiment {experiment.name}')
                print("Unexpected error:", sys.exc_info())
                print('Trying to execute remaining experiment_scripts')
                if mlflow.active_run():
                    mlflow.end_run()

            finally:
                experiment.cleanup()
                print("\n")
