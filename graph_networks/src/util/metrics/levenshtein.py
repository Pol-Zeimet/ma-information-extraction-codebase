import mlflow
from Levenshtein import distance as levenshtein_distance
from statistics import mean

def postprocess_tokens(tokens_in: list, label_type: str):

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

        if len(true_org_text) > 0:
            org_text_distance = levenshtein_distance(org_text, true_org_text)
            org_distances.append(org_text_distance)
            org_coverages.append((len(true_org_text) - org_text_distance) / len(true_org_text) * 100)
        if len(true_addr_text) > 0:
            addr_text_distance = levenshtein_distance(addr_text, true_addr_text)
            addr_distances.append(addr_text_distance)
            addr_coverages.append((len(true_addr_text) - addr_text_distance) / len(true_addr_text) * 100)
        if len(true_date_text) > 0:
            date_text_distance = levenshtein_distance(date_text, true_date_text)
            date_distances.append(date_text_distance)
            date_coverages.append((len(true_date_text) - date_text_distance) / len(true_date_text) * 100)
        if len(true_total_text) > 0:
            total_text_distance = levenshtein_distance(total_text, true_total_text)
            total_distances.append(total_text_distance)
            total_coverages.append((len(true_total_text) - total_text_distance) / len(true_total_text) * 100)

    mean_addr_distances = mean(addr_distances)
    mean_org_distances = mean(org_distances)
    mean_total_distances = mean(total_distances)
    mean_date_distances = mean(date_distances)
    total_mean = (mean_addr_distances + mean_org_distances + mean_total_distances + mean_date_distances) / 4

    mean_addr_coverages = mean(addr_coverages)
    mean_org_coverages = mean(org_coverages)
    mean_total_coverages = mean(total_coverages)
    mean_date_coverages = mean(date_coverages)
    mean_coverage = (mean_addr_coverages + mean_org_coverages + mean_total_coverages + mean_date_coverages) / 4

    if training:
        status = 'train'
    else:
        status = 'eval'

        
    mlflow.log_metric(f'{status}_mean_addr_distances', mean_addr_distances)
    mlflow.log_metric(f'{status}_mean_org_distances', mean_org_distances)
    mlflow.log_metric(f'{status}_mean_total_distances', mean_total_distances)
    mlflow.log_metric(f'{status}_mean_date_distances', mean_date_distances)
    mlflow.log_metric(f'{status}_total_mean', total_mean)
    print("Latest levenshtein:")
    print(f"{status} addr distances: {mean_addr_distances}")
    print(f"{status} org distances: {mean_org_distances}")
    print(f"{status} 'total' distances: {mean_total_distances}")
    print(f"{status} date distances: {mean_date_distances}")
    print(f"Average: {total_mean}")

    mlflow.log_metric(f'{status}_mean_addr_coverage', mean_addr_coverages)
    mlflow.log_metric(f'{status}_mean_org_coverage', mean_org_coverages)
    mlflow.log_metric(f'{status}_mean_total_coverage', mean_total_coverages)
    mlflow.log_metric(f'{status}_mean_date_coverage', mean_date_coverages)
    mlflow.log_metric(f'{status}_mean_coverage', mean_coverage)
    print("Coverage:")
    print(f'{status}_mean_addr_coverage:  {mean_addr_coverages}')
    print(f'{status}_mean_org_coverage:  {mean_org_coverages}')
    print(f'{status}_mean_total_coverage:  {mean_total_coverages}')
    print(f'{status}_mean_date_coverage:  {mean_date_coverages}')
    print(f'{status}_mean_coverage: {mean_coverage}')