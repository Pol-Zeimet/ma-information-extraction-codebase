from datetime import datetime
from typing import List

from src import BaseExperiment


class ExperimentPipeline:
    def run(self, experiments: List[BaseExperiment]):
        for i, experiment in enumerate(experiments):
            try:
                print(f"{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}  Experiment {i+1}/{len(experiments)}: {experiment.name}")
                experiment.setup()
                experiment.run()
            except:
                print('something broke')

            finally:
                experiment.cleanup()
                print("\n")