from datetime import datetime
from typing import List

from src.experiments.experiment_class.experiment_class import BaseExperiment


class ExperimentPipeline:
    def run(self, experiments: List[BaseExperiment]):
        for i, experiment in enumerate(experiments):
            print(f"{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}  Experiment {i+1}/{len(experiments)}: {experiment.name}")
            experiment.setup()
            experiment.run()
            experiment.cleanup()
            print("\n")
