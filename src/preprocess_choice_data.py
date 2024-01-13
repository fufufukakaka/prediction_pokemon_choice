import random

import cloudpickle
import pandas as pd


def main():
    df = pd.read_csv("data/pokemon_selection/poke_battle_logger_pokemon_selection.csv")
    your_teams = df["your_team"].values.tolist()
    opponent_teams = df["opponent_team"].values.tolist()
    opponent_choice_1 = df["opponent_pokemon_1"].values.tolist()
    opponent_choice_2 = df["opponent_pokemon_2"].values.tolist()
    opponent_choice_3 = df["opponent_pokemon_3"].values.tolist()

    # shuffle してデータを増やす
    augmented_your_teams = []
    augmented_opponent_teams = []
    augmented_choices = []
    for _ in range(10):
        for your_team, opponent_team, _choice_1, _choice_2, _choice_3 in zip(your_teams, opponent_teams, opponent_choice_1, opponent_choice_2, opponent_choice_3):
            augmented_your_teams.append(random.sample(your_team.split(","), 6))
            augmented_opponent_teams.append(random.sample(opponent_team.split(","), 6))
            augmented_choices.append([_choice_1, _choice_2, _choice_3])

    concat_data = []
    for your_team, opponent_team, choice in zip(augmented_your_teams, augmented_opponent_teams, augmented_choices):
        not_choices_oppoents = [x for x in opponent_team if x not in choice]
        for index, _choice in enumerate(choice):
            if _choice == "Unseen":
                continue
            if index == 0:
                concat_data.append(
                    {
                        "text": " ".join(your_team) + "[SEP]" + " ".join(opponent_team) + "[SEP]" + _choice,
                        "label": "first_choiced"
                    }
                )
            else:
                concat_data.append(
                    {
                        "text": " ".join(your_team) + "[SEP]" + " ".join(opponent_team) + "[SEP]" + _choice,
                        "label": "choiced"
                    }
                )
        for _choice in not_choices_oppoents:
            concat_data.append(
                {
                    "text": " ".join(your_team) + "[SEP]" + " ".join(opponent_team) + "[SEP]" + _choice,
                    "label": "not_choiced"
                }
            )

    cloudpickle.dump(concat_data, open("data/pokemon_selection/poke_battle_logger_pokemon_selection_preprocessed.pkl", "wb"))

if __name__ == "__main__":
    main()
