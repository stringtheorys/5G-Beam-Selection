
import csv
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm


def tensorboard_to_csv():
    # csv: model name, setting, metric, step, value
    with open('../results/tensorboard-logs.csv', 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['model-name', 'setting', 'metric', 'step', 'value'])
        writer.writeheader()

        folder = '../results/logs/'
        model_names = os.listdir(folder)
        print(model_names)
        for model_name in tqdm(model_names):
            model_folder = os.listdir(f'{folder}/{model_name}')[0]
            print(model_name, model_folder)

            if 'federated' in model_name:
                for vehicle in os.listdir(f'{folder}/{model_name}/{model_folder}'):
                    print(vehicle)
                    for setting in ['train', 'validation']:
                        event_acc = EventAccumulator(f'{folder}/{model_name}/{model_folder}/{vehicle}/{setting}').Reload()
                        metrics = event_acc.Tags()['scalars']
                        for metric in metrics:
                            data = event_acc.Scalars(metric)
                            metric = metric.replace('epoch_', '')
                            for event in data:
                                writer.writerow({'model-name': f'{model_name}-{vehicle}', 'setting': setting,
                                                 'metric': metric, 'step': event.step, 'value': event.value})
            else:
                for setting in ['train', 'validation']:
                    event_acc = EventAccumulator(f'{folder}/{model_name}/{model_folder}/{setting}').Reload()
                    metrics = event_acc.Tags()['scalars']
                    for metric in metrics:
                        data = event_acc.Scalars(metric)
                        metric = metric.replace('epoch_', '')
                        for event in data:
                            writer.writerow({'model-name': model_name, 'setting': setting, 'metric': metric,
                                             'step': event.step, 'value': event.value})
    print('Finished')


tensorboard_to_csv()
