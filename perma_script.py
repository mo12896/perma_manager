import os
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.figure import Figure
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Settings
pd.set_option("display.max_columns", 6)
# Constants
IMG_DIR = Path.cwd() / "perma_scores"
# Define the group ids and the date for which you want to fetch the data
group_ids = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
]
# year-month-day
date = "2023-01-10"
send = False
last_day = False
aggregate_func = np.mean
sender = "moritz96@mit.edu"
smtp_server = "outgoing.mit.edu"


def login_to_gdrive() -> GoogleDrive:
    # Initializing a GoogleAuth Object
    gauth = GoogleAuth()
    # client_secrets.json file is verified
    # and it automatically handles authentication
    gauth.LocalWebserverAuth()
    gauth.Authorize()
    # GoogleDrive Instance is created using
    # authenticated GoogleAuth instance
    gdrive = GoogleDrive(gauth)

    return gdrive


def read_file_from_gdrive(gdrive: GoogleDrive, file_id: str) -> None:
    # Initialize GoogleDriveFile instance with file id
    file_obj = gdrive.CreateFile({"id": file_id})
    file_obj.GetContentFile(
        "perma_data.xls",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def compute_perma(data: pd.DataFrame, aggregate_func: Callable) -> pd.DataFrame:
    return data.apply(aggregate_func, axis=0)


def join_two_dataframes(
    df1: pd.DataFrame, df2: pd.DataFrame, key: str = "E-Mail-Adresse"
) -> pd.DataFrame:
    return df1.join(df2.set_index(key), on=key)


def extract_date_from_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = [
        timestamp.split(" ")[0] for timestamp in df["Zeitstempel"].astype("string")
    ]
    return df


def get_email_recipients(df: pd.DataFrame, group_id: int) -> list[str]:
    emails = df[df["Group_IDs"] == group_id]["E-Mail-Adresse"].unique()
    return [str(email) for email in emails]


def rename_df_columns(df: pd.DataFrame) -> None:
    df.rename(
        columns={
            "1. I feel joy in a typical workday": "1",
            "2. Overall, I feel enthusiastic about my work": "2",
            "3. I love my job": "3",
            "4. I typically become absorbed while I am working on \nsomething that challenges my abilities": "4",
            "5. I lose track of time while doing something I enjoy at work": "5",
            "6. When I am working on something I enjoy, I forget everything else around me": "6",
            "7. I can receive support from coworkers if I need it": "7",
            "8. I feel appreciated by my coworkers": "8",
            "9. I trust my colleagues": "9",
            "10. My colleagues bring out my best self": "10",
            "11. My work is meaningful": "11",
            "12. I understand what makes my job meaningful": "12",
            "13. The work I do serves a greater purpose": "13",
            "14. I set goals that help me achieve my career aspirations": "14",
            "15. I typically accomplish what I set out to do in my job": "15",
            "16. I am generally satisfied with my performance at work": "16",
        },
        inplace=True,
    )


def aggregate_perma_factors(df: pd.DataFrame) -> list:
    perma_df = []
    scores = (
        [df["1"], df["2"], df["3"]],
        [df["4"], df["5"], df["6"]],
        [df["7"], df["8"], df["9"], df["10"]],
        [df["11"], df["12"], df["13"]],
        [df["14"], df["15"], df["16"]],
    )
    for score in scores:
        perma_df.append(round(sum(score) / len(score), 2))
    return perma_df


def save_perma_scores(
    perma_score: list,
    group_id: str,
    date: str,
    filename: Path = Path("perma_scores.csv"),
) -> None:
    def key_exists(key: str, filename: Path = Path("perma_scores.csv")) -> bool:
        df = pd.read_csv(filename)
        return df["key"].isin([key]).any()

    key = f"{group_id}_{date}"
    data = pd.DataFrame(
        {
            "key": key,
            "group_id": group_id,
            "date": date,
            "values": [perma_score],
        }
    )
    if not filename.exists():
        data.to_csv(filename, mode="w", index=False, header=True)
        return
    if not key_exists(key):
        data.to_csv(filename, mode="a", index=False, header=False)


def read_perma_scores(
    group_ids: list[str], filename: Path = Path("perma_scores.csv")
) -> list[pd.DataFrame]:
    df = pd.read_csv(filename)
    return [df.loc[df["group_id"] == group_id] for group_id in group_ids]


def send_mail(
    image_files: list[Path], recipients: list[str], sender: str = sender
) -> None:
    # Create the container (outer) email message.
    msg = MIMEMultipart()
    msg["Subject"] = "This is your Team PERMA score for today!"
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.preamble = "PERMA scores"

    # Create the body of the message and attach to the container
    with open("perma.txt", "r") as f:
        email_msg = f.read()
    body = MIMEText(email_msg)
    msg.attach(body)

    # Open the files in binary mode.  Let the MIMEImage class automatically
    # guess the specific image type.
    for img in image_files:
        with open(img, "rb") as fp:
            img = MIMEImage(fp.read())
            img.add_header(
                "Content-Disposition", "attachment", filename="team_perma.png"
            )
        msg.attach(img)

    # Send the email via the MIT SMTP server
    # (http://kb.mit.edu/confluence/pages/viewpage.action?pageId=155282952)
    try:
        s = smtplib.SMTP(host=smtp_server)
        s.sendmail(sender, recipients, msg.as_string())
        s.quit()
        print(f"Successfully sent email with image: {image_files}!")
    except smtplib.SMTPException as exc:
        print("Error sending email!")
        print(exc)


def create_bar_plot(df: pd.DataFrame, title: str) -> Union[Figure, None]:
    labels = ("P", "E", "R", "M", "A")
    x = np.arange(1, 6)
    size = df.shape[0]
    width = 1 / (size * 2)

    fig, ax = plt.subplots()

    for index, row in enumerate(df.itertuples()):
        values = [float(i) for i in row.values[1:-1].split(", ")]

        # Check if there are only nan values in the data
        values_check = np.array([np.isnan(i) for i in values])
        if values_check.all():
            return None

        rects = ax.bar(
            x - width * (size - index - 1), values, width, label=f"{row.date}"
        )

        ax.bar_label(rects, padding=3)
        ax.plot(np.arange(1, 6), [7] * 5, color="red", lw=2)
        ax.text(4, 7.05, "Upper Limit", color="red")

    ax.set_title(title)
    ax.set_ylabel("PERMA Score")
    ax.set_xticks(x, labels)
    ax.legend()

    return fig


def create_line_plot(df: pd.DataFrame, title: str) -> Figure:
    def create_axhline(
        values: np.ndarray,
        lowerbound: int,
        upperbound: int,
        text: str,
        color: str = "green",
    ) -> None:
        mean = values[lowerbound:upperbound].mean()

        plt.text(
            lowerbound + 1,
            mean,
            text + str(round(mean, 2)),
        )
        ax.axhline(
            mean,
            lowerbound / 16,
            upperbound / 16,
            color=color,
            lw=2,
            alpha=0.7,
        )

    fig, ax = plt.subplots()

    x = np.arange(1, len(df.values) + 1)
    y = df.values
    ax.plot(x, y)

    create_axhline(y, 0, 16, text="Average Score: ", color="black")
    create_axhline(y, 0, 3, text="P-Score: ")
    create_axhline(y, 3, 7, text="E-Score: ")
    create_axhline(y, 7, 10, text="R-Score: ")
    create_axhline(y, 10, 14, text="M-Score: ")
    create_axhline(y, 14, 16, text="A-Score: ")

    ax.set_xlim(1, 16)
    ax.set_ylim(0, 8)
    ax.set_title(title)
    ax.set_xlabel("PERMA Question")
    ax.set_ylabel("PERMA Score")
    ax.grid(True)

    return fig


def create_box_plot(dataframes: list[pd.DataFrame], title: str) -> Figure:
    def read_cohort_permas(dataframes: list[pd.DataFrame]) -> np.ndarray:
        cohort_permas = []
        for df in dataframes:
            group_permas = []
            for _, row in enumerate(df.itertuples()):
                values = [float(i) for i in row.values[1:-1].split(", ")]
                group_permas.append(values)
            cohort_permas.append(group_permas)

        return np.array(cohort_permas)

    def extract_daily_cohort_perma(cohort_permas: np.ndarray, row: int) -> list:
        daily_cohort_perma = []
        for i in range(5):
            perma_factor = []
            for group in cohort_permas:
                perma_factor.append(group[row, i])
            daily_cohort_perma.append(perma_factor)
        return daily_cohort_perma

    fig, ax = plt.subplots()

    labels = ("P", "E", "R", "M", "A")
    cohort_permas = read_cohort_permas(dataframes)

    # TODO: Enable this for 3 days!
    data_0 = extract_daily_cohort_perma(cohort_permas, 0)
    data_1 = extract_daily_cohort_perma(cohort_permas, 1)
    # data_2 = get_daily_cohort_perma(cohort_permas, 2)

    ax.boxplot(
        data_0, positions=np.array(range(len(data_0))) * 2.0 - 0.4, sym="", widths=0.6
    )
    ax.boxplot(
        data_1, positions=np.array(range(len(data_1))) * 2.0 + 0.4, sym="", widths=0.6
    )
    # ax.boxplot(data_2)

    ax.set_title(title)
    ax.set_xticks(range(0, len(labels) * 2, 2), labels)
    ax.set_ylabel("PERMA Score")

    return fig


def save_figure(fig: Figure, group_id: str, foldername: Path) -> None:
    if not foldername.exists():
        foldername.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(foldername / f"{group_id}.png"))


def main():

    load_dotenv()

    # load the data
    if not Path("perma_data.xls").exists():
        gdrive = login_to_gdrive()
        read_file_from_gdrive(gdrive, str(os.getenv("FILE_ID")))

    perma_df = pd.read_excel("perma_data.xls")
    group_id_df = pd.read_excel("group_ids.xls")

    # Preprocess the data
    df = join_two_dataframes(perma_df, group_id_df)
    df = extract_date_from_timestamp(df)
    rename_df_columns(df)

    # Compute Team PERMA and create plot
    for g_id in group_ids:
        group_id = f"group_{g_id}"

        # Note: select the correct column indices from the table!
        team_perma = compute_perma(
            data=df[(df["Group_IDs"] == g_id) & (df["Date"] == date)].iloc[:, 6:22],
            aggregate_func=aggregate_func,
        )
        team_perma = aggregate_perma_factors(team_perma)

        save_perma_scores(team_perma, group_id, date)
        overall_team_perma = read_perma_scores([group_id])[0]

        fig = create_bar_plot(
            overall_team_perma,
            title=f"Your overall Team PERMA score on {date}",
        )

        if fig:
            folder = IMG_DIR / f"{date}"
            save_figure(fig, group_id, folder)

    # Send email to the recipients
    if send:
        for g_id in group_ids:
            team_perma = IMG_DIR / date / f"group_{g_id}.png"

            if team_perma.exists():
                send_mail(
                    image_files=[team_perma],
                    recipients=get_email_recipients(df, g_id),
                )

    if last_day:
        overall_cohort_perma = read_perma_scores(
            [f"group_{g_id}" for g_id in group_ids]
        )
        fig = create_box_plot(
            overall_cohort_perma,
            title=f"The overall cohort PERMA score during the Intensive Week 2023",
        )
        folder = Path.cwd()
        save_figure(fig, "cohort_perma", folder)


if __name__ == "__main__":
    main()
