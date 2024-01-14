import random

import cloudpickle
import pandas as pd
from sklearn.model_selection import train_test_split


def build_data(df, is_train=True):
    your_teams = df["your_team"].apply(lambda x: x.split(",")).values.tolist()
    opponent_teams = df["opponent_team"].apply(lambda x: x.split(",")).values.tolist()
    opponent_choice_1 = df["opponent_pokemon_1"].values.tolist()
    opponent_choice_2 = df["opponent_pokemon_2"].values.tolist()
    opponent_choice_3 = df["opponent_pokemon_3"].values.tolist()

    # shuffle してデータを増やす
    if is_train:
        augmented_your_teams = []
        augmented_opponent_teams = []
        augmented_choices = []
        for _ in range(10):
            for your_team, opponent_team, _choice_1, _choice_2, _choice_3 in zip(your_teams, opponent_teams, opponent_choice_1, opponent_choice_2, opponent_choice_3):
                augmented_your_teams.append(random.sample(your_team, 6))
                augmented_opponent_teams.append(random.sample(opponent_team, 6))
                augmented_choices.append([_choice_1, _choice_2, _choice_3])

    concat_data = []
    if is_train:
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
    else:
        for your_team, opponent_team, _choice_1, _choice_2, _choice_3 in zip(your_teams, opponent_teams, opponent_choice_1, opponent_choice_2, opponent_choice_3):
            choice = [_choice_1, _choice_2, _choice_3]
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

    return concat_data


def main():
    df = pd.read_csv("data/pokemon_selection/poke_battle_logger_pokemon_selection.csv")
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, validation_df = train_test_split(train_df, test_size=0.1, random_state=42)

    train_concat_data = build_data(train_df)
    validation_concat_data = build_data(validation_df, is_train=False)
    test_concat_data = build_data(test_df, is_train=False)

    cloudpickle.dump(train_concat_data, open("data/pokemon_selection/train_poke_battle_logger_pokemon_selection_preprocessed.pkl", "wb"))
    cloudpickle.dump(validation_concat_data, open("data/pokemon_selection/validation_poke_battle_logger_pokemon_selection_preprocessed.pkl", "wb"))
    cloudpickle.dump(test_concat_data, open("data/pokemon_selection/test_poke_battle_logger_pokemon_selection_preprocessed.pkl", "wb"))

if __name__ == "__main__":
    main()
