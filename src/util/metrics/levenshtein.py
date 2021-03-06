import mlflow
from Levenshtein import distance as levenshtein_distance
from statistics import mean


def postprocess_tokens(tokens_in: list, label_type: str):
    """
    Remove unwanted characters from strings. This includes ##, ' ' for all labels, aswell as ',' '.' and ':' if the label of the string is not "DATE" or "MONEY"
    :param tokens_in: the string to process
    :param label_type: the label of the string
    :return:
    """
    string_out = ''.join(tokens_in)
    string_out = string_out.replace('##', '') \
                           .replace(' ', '') \
                           .strip()

    if label_type != 'date' and label_type != 'total':
        string_out = string_out.replace(',', '') \
            .replace('.', '')\
            .replace(':', '')

    return str.lower(string_out)


def compute_levenshtein(y_pred, y_true, token_lists, training: bool):
    """
    Compute the Levenshtein distance and coverage betweeen ground truth and extracted token.
    Logs the metric with MlFlow.
    :param y_pred: A batch of Lists of predicted labels
    Labels are given as integers
    :param y_true: A batch of Lists of true labels
    Labels are given as integers
    :param token_lists: a batch of lists with tokens
    :param training: boolean value to indicate if this is a training or evaluation run. only useful for logging
    :return:
    """
    if training:
        status = 'train'
    else:
        status = 'eval'

    addr_distances = []
    org_distances = []
    total_distances = []
    date_distances = []
    addr_coverages = []
    org_coverages = []
    total_coverages = []
    date_coverages = []


    for pred, true, token_list in zip(y_pred, y_true, token_lists):
        true_addr, true_org, true_total, true_date = [], [], [], []
        addr, org, total, date = [], [], [], []
        for idx, (pred_label, true_label) in enumerate(zip(pred, true)):
            if pred_label == 1:
                total.append(token_list[idx])
            elif pred_label == 2:
                org.append(token_list[idx])
            elif pred_label == 3:
                date.append(token_list[idx])
            elif pred_label == 4:
                addr.append(token_list[idx])

            if true_label == 1:
                true_total.append(token_list[idx])
            elif true_label == 2:
                true_org.append(token_list[idx])
            elif true_label == 3:
                true_date.append(token_list[idx])
            elif true_label == 4:
                true_addr.append(token_list[idx])

        true_addr_text = postprocess_tokens(true_addr, 'addr')
        true_org_text = postprocess_tokens(true_org, 'org')
        true_total_text = postprocess_tokens(true_total, 'total')
        true_date_text = postprocess_tokens(true_date, 'date')

        addr_text = postprocess_tokens(addr, 'addr')
        org_text = postprocess_tokens(org, 'org')
        total_text = postprocess_tokens(total, 'total')
        date_text = postprocess_tokens(date, 'date')

        calculate_coverage(addr_coverages, addr_distances, addr_text, date_coverages, date_distances, date_text,
                           org_coverages, org_distances, org_text, total_coverages, total_distances, total_text,
                           true_addr_text, true_date_text, true_org_text, true_total_text)

    calculate_mean_metric_and_log(addr_coverages, addr_distances, date_coverages, date_distances, org_coverages,
                                  org_distances, status, total_coverages, total_distances)


def calculate_mean_metric_and_log(addr_coverages, addr_distances, date_coverages, date_distances, org_coverages,
                                  org_distances, status, total_coverages, total_distances):
    """
    calculate mean metrics over the given batch and log them using mlflow
    :param addr_coverages:
    :param addr_distances:
    :param date_coverages:
    :param date_distances:
    :param org_coverages:
    :param org_distances:
    :param status: training or testing/evaluating
    :param total_coverages:
    :param total_distances:
    :return:
    """
    all_distance_means = []
    all_coverage_means = []
    print("Latest levenshtein and Coverage:")
    if len(addr_distances) > 0:
        mean_addr_distances = mean(addr_distances)
        mean_addr_coverages = mean(addr_coverages)
        all_distance_means.append(mean_addr_distances)
        all_coverage_means.append(mean_addr_coverages)

        mlflow.log_metric(f'{status}_mean_addr_distances', mean_addr_distances)
        mlflow.log_metric(f'{status}_mean_addr_coverage', mean_addr_coverages)
        print(f"{status} addr distances: {mean_addr_distances}")
        print(f'{status}_mean_addr_coverage:  {mean_addr_coverages}')
    if len(org_distances) > 0:
        mean_org_distances = mean(org_distances)
        mean_org_coverages = mean(org_coverages)
        all_distance_means.append(mean_org_distances)
        all_coverage_means.append(mean_org_coverages)

        mlflow.log_metric(f'{status}_mean_org_distances', mean_org_distances)
        mlflow.log_metric(f'{status}_mean_org_coverage', mean_org_coverages)
        print(f'{status}_mean_org_coverage:  {mean_org_coverages}')
        print(f"{status} org distances: {mean_org_distances}")
    if len(total_distances) > 0:
        mean_total_distances = mean(total_distances)
        mean_total_coverages = mean(total_coverages)
        all_distance_means.append(mean_total_distances)
        all_coverage_means.append(mean_total_coverages)

        mlflow.log_metric(f'{status}_mean_total_distances', mean_total_distances)
        mlflow.log_metric(f'{status}_mean_total_coverage', mean_total_coverages)
        print(f'{status}_mean_total_coverage:  {mean_total_coverages}')
        print(f"{status} 'total' distances: {mean_total_distances}")
    if len(date_distances) > 0:
        mean_date_distances = mean(date_distances)
        mean_date_coverages = mean(date_coverages)
        all_distance_means.append(mean_date_distances)
        all_coverage_means.append(mean_date_coverages)

        mlflow.log_metric(f'{status}_mean_date_distances', mean_date_distances)
        mlflow.log_metric(f'{status}_mean_date_coverage', mean_date_coverages)
        print(f'{status}_mean_date_coverage:  {mean_date_coverages}')
        print(f"{status} date distances: {mean_date_distances}")
    if len(all_distance_means) > 0:
        total_mean = mean(all_distance_means)
        mlflow.log_metric(f'{status}_total_mean', total_mean)

        mean_coverage = mean(all_coverage_means)
        mlflow.log_metric(f'{status}_mean_coverage', mean_coverage)
        print(f'{status}_mean_coverage: {mean_coverage}')
        print(f"Average: {total_mean}")


def calculate_coverage(addr_coverages, addr_distances, addr_text, date_coverages, date_distances, date_text,
                       org_coverages, org_distances, org_text, total_coverages, total_distances, total_text,
                       true_addr_text, true_date_text, true_org_text, true_total_text):
    """Calculate Coverage in one element of the batch for each label.
        Uses the length of the truth and extracted text aswell as their levenshtein distance.
        Coverage can both be over 100% or 0% if either too much is extracted or nothing matches.
        In most observed cases, a value over 100%percent means a very bad extraction.
    """
    if len(true_org_text) > 0:
        org_text_distance = levenshtein_distance(org_text, true_org_text)
        org_distances.append(org_text_distance)
        if len(true_org_text) > org_text_distance:
            org_coverages.append((len(true_org_text) - org_text_distance) / len(true_org_text) * 100)
        else:
            org_coverages.append((org_text_distance - len(true_org_text)) / len(true_org_text) * 100)
    if len(true_addr_text) > 0:
        addr_text_distance = levenshtein_distance(addr_text, true_addr_text)
        addr_distances.append(addr_text_distance)
        if len(true_addr_text) > addr_text_distance:
            addr_coverages.append((len(true_addr_text) - addr_text_distance) / len(true_addr_text) * 100)
        else:
            addr_coverages.append((addr_text_distance - len(true_addr_text)) / len(true_addr_text) * 100)
    if len(true_date_text) > 0:
        date_text_distance = levenshtein_distance(date_text, true_date_text)
        date_distances.append(date_text_distance)
        if len(true_date_text) > date_text_distance:
            date_coverages.append((len(true_date_text) - date_text_distance) / len(true_date_text) * 100)
        else:
            date_coverages.append((date_text_distance - len(true_date_text)) / len(true_date_text) * 100)
    if len(true_total_text) > 0:
        total_text_distance = levenshtein_distance(total_text, true_total_text)
        total_distances.append(total_text_distance)
        if len(true_total_text) > total_text_distance:
            total_coverages.append((len(true_total_text) - total_text_distance) / len(true_total_text) * 100)
        else:
            total_coverages.append((total_text_distance - len(true_total_text)) / len(true_total_text) * 100)


