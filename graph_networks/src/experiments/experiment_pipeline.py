from datetime import datetime
from typing import List

from src.experiments.experiment_class.experiment_class import BaseExperiment


class ExperimentPipeline:
    def run(self, experiments: List[BaseExperiment]):
        for i, experiment in enumerate(experiments):
            try:
                print(
                    f"{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}  Experiment {i + 1}/{len(experiments)}: {experiment.name}")
                experiment.setup()
                experiment.run()

            except RuntimeError as err:
                print(f'An error occurred while executing experiment {experiment.name}')
                print(err)
                print('Trying to execute remaining experiments')

            finally:
                experiment.cleanup()
                print("\n")
