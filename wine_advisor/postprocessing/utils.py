from leven import levenshtein
from pylcs import lcs_string_length

def find_closest_match(input_text, choices, threshold=2):
    """
    Trouve le match le plus proche parmi une liste de mots en utilisant la distance de Levenshtein.
    Renvoie la correspondance la plus proche si elle se situe en dessous du seuil, sinon None.
    """

    closest_match = None
    min_distance = float("inf")

    for choice in choices:
        distance = levenshtein(input_text, choice)
        if distance < min_distance:
            min_distance = distance
            closest_match = choice
            # print(input_text, choice, min_distance)
    return closest_match if min_distance <= threshold else None


def find_lcs(line, data):
    closest_match = ''
    min_distance = 0

    for choice in data:
        distance = lcs_string_length(choice, line)/len(choice)
        if distance == min_distance:
            closest_match = choice if len(choice) > len(closest_match) else closest_match
        elif distance > min_distance:
            min_distance = distance
            closest_match = choice
            # print(line, choice, min_distance)
    return (closest_match, min_distance, line.lower())


if __name__ == '__main__':
    from wine_advisor.loading_resources import load_csv
    data = load_csv('../tmp/resources/appellation_classification.csv')
    print(find_lcs('haut medoc', data))
