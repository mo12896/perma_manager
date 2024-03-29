import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.figure import Figure


def standardize_dataframe(df, columns=None):
    """
    Standardizes selected columns of a dataframe using the mean and standard deviation
    of all rows.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to be standardized.
    columns : list or None, optional
        The list of column names to standardize. If None, standardizes all columns.
        Default is None.

    Returns:
    --------
    pandas.DataFrame
        The standardized dataframe.
    """
    if columns is None:
        columns = df.columns

    # calculate the mean and standard deviation of all rows for selected columns
    mean = df[columns].mean(axis=0)
    std = df[columns].std(axis=0)

    # standardize selected columns using the mean and standard deviation of all rows
    df_standardized = df.copy()
    df_standardized[columns] = (df[columns] - mean) / std

    return df_standardized


def plot_daily_email_counts(df):
    grouped = df.groupby("Day")["E-Mail-Adresse"].value_counts()

    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 6))
    grouped.unstack().plot(kind="barh", ax=ax)
    ax.set_xlabel("Day")
    ax.set_ylabel("Count")
    plt.show()


def plot_total_email_counts(df):
    email_counts = df["E-Mail-Adresse"].value_counts()
    email_counts.plot(kind="barh")
    plt.show()
    
    
def plot_perma(dataframe: pd.DataFrame, title: str) -> Figure:
    fig, ax = plt.subplots()

    labels = ["P", "E", "R", "M", "A"]
    days = ["Day1", "Day2", "Day3", "Day4"]
    colors = ["pink", "lightblue", "lightgreen", "lightyellow"]
    offsets = [0.45, 0.15, -0.15, -0.45]

    for i, day in enumerate(range(10, 14)):
        data = []
        for label in labels:
            data.append(dataframe[dataframe.Day == day][label].values)
        ax.boxplot(
            data,
            positions=np.array(range(len(data))) * 2.0 - offsets[i],
            sym="",
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor=colors[i]),
            medianprops=dict(color="black"),
        )

    ax.set_title(title)
    ax.set_xticks(range(0, len(labels) * 2, 2), labels)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.set_ylabel("PERMA Score")

    handles = [
        patches.Patch(color=color, label=label) for label, color in zip(days, colors)
    ]

    ax.legend(handles=handles, loc="lower right", fontsize=10)
    plt.show()


if __name__ == "__main__":
    WEEK = False
    # Read in the CSV file
    df1 = pd.read_csv("PERMA_scores.csv")
    df2 = pd.read_csv("couhes_list.csv")

    df = df1[df1["E-Mail-Adresse"].isin(df2["E-Mail-Adresse"])]

    # Add new column
    df["Day"] = pd.to_datetime(
        df1["Zeitstempel"].str.split().str[0], format="%d.%m.%Y"
    ).dt.day

    # Drop unnecessary columns
    df = df.drop(
        columns=[
            "Zeitstempel",
            "First Name",
            "Last Name/Surname",
            "To which day do your answers relate to? ",
            "From time to time, most people discuss important personal matters with other people. Looking back over the last six months – that would be back to last July – who are the people with whom you discussed an important personal matter?",
            "Accomplish team goals.",
            "Meet performance goals in a timely manner.",
            "Achieve team goals with few or no errors.",
        ]
    )

    # Rename columns
    df = df.rename(
        columns={
            "1. I feel joy in a typical workday": "P1",
            "2. Overall, I feel enthusiastic about my work": "P2",
            "3. I love my job": "P3",
            "4. I typically become absorbed while I am working on \nsomething that challenges my abilities": "E1",
            "5. I lose track of time while doing something I enjoy at work": "E2",
            "6. When I am working on something I enjoy, I forget everything else around me": "E3",
            "7. I can receive support from coworkers if I need it": "R1",
            "8. I feel appreciated by my coworkers": "R2",
            "9. I trust my colleagues": "R3",
            "10. My colleagues bring out my best self": "R4",
            "11. My work is meaningful": "M1",
            "12. I understand what makes my job meaningful": "M2",
            "13. The work I do serves a greater purpose": "M3",
            "14. I set goals that help me achieve my career aspirations": "A1",
            "15. I typically accomplish what I set out to do in my job": "A2",
            "16. I am generally satisfied with my performance at work": "A3",
        }
    )

    # Define the pillars of PERMA columns to be averaged
    columns_dict = {
        "P": ["P1", "P2", "P3"],
        "E": ["E1", "E2", "E3"],
        "R": ["R1", "R2", "R3", "R4"],
        "M": ["M1", "M2", "M3"],
        "A": ["A1", "A2", "A3"],
    }

    # compute the mean for each pillar of columns and store the result in a new column
    for letter, columns in columns_dict.items():
        df[letter] = df.loc[:, columns].mean(axis=1)

    # Either stndardize based off whole week or per day
    if WEEK:
        df = standardize_dataframe(df, list(columns_dict.keys()))
    else:
        df = df.groupby("Day", group_keys=False).apply(
            standardize_dataframe, columns=list(columns_dict.keys())
        )

    # Drop unnecessary columns
    df = df.drop(
        columns=[
            "P1",
            "P2",
            "P3",
            "E1",
            "E2",
            "E3",
            "R1",
            "R2",
            "R3",
            "R4",
            "M1",
            "M2",
            "M3",
            "A1",
            "A2",
            "A3",
        ]
    )

    # merge the dataframes based on the email address to get participants names
    df_merged = pd.merge(df, df2, on="E-Mail-Adresse", how="left")

    # re-order the columns
    df_merged = df_merged[
        [
            "E-Mail-Adresse",
            "First Name",
            "Last Name/Surname",
            "Day",
            "P",
            "E",
            "R",
            "M",
            "A",
        ]
    ]

    df_merged.to_csv("perma_scores_dataset.csv", index=False)
    print(df_merged)

    plot_total_email_counts(df_merged)
    plot_daily_email_counts(df_merged)
    plot_perma(df_merged, "PERMA Scores")
