from datetime import datetime
from typing import List
import mlflow
import sys
import os
from src.experiments.experiment_class.experiment_class import BaseExperiment


class ExperimentPipeline:
    @staticmethod
    def run(experiments: List[BaseExperiment], filename):
        for i, experiment in enumerate(experiments):
            try:
                print(
                    f"{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}  Experiment {i + 1}/{len(experiments)}: {experiment.name}")
                experiment.setup()
                experiment.run()

            except:
                print(f'An error occurred while executing experiment {experiment.name}')
                print(f"writing to file {os.path.abspath(filename)}")
                with open(filename, 'w') as f:
                    f.write(datetime.today().strftime('%d-%m-%Y-%H:%M:%S'))
                    f.write('\n')
                    f.write(f'Error in Experiment run for {experiment.name} \n')
                    f.write(str(sys.exc_info()))
                    f.write('\n')
                    f.close()
                print('Trying to execute remaining experiment_scripts')

                if mlflow.active_run():
                    mlflow.end_run()

            finally:
                experiment.cleanup()
                print("\n")
