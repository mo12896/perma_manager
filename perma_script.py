import os
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Settings
pd.set_option("display.max_columns", 6)
# Constants
IMG_DIR = Path.cwd() / "perma_scores"
# Define the group ids and the days for which you want to fetch the data
group_ids = [1]
dates = ["2022-12-12"]
send: bool = False
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


def compute_team_perma(df: pd.DataFrame, aggregate_func: Callable) -> pd.DataFrame:
    return df.iloc[:, 4:20].apply(aggregate_func, axis=0)


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


def send_mail(image_file: Path, recipients: list[str], sender: str = sender) -> None:
    # Create the container (outer) email message.
    msg = MIMEMultipart()
    msg["Subject"] = "This is your Teams's PERMA score for today!"
    # me == the sender's email address
    # recipients = the list of all recipients' email addresses
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.preamble = "Team PERMA score"

    # Open the files in binary mode.  Let the MIMEImage class automatically
    # guess the specific image type.
    with open(image_file, "rb") as fp:
        img = MIMEImage(fp.read())
    msg.attach(img)

    # Send the email via the MIT SMTP server
    # (http://kb.mit.edu/confluence/pages/viewpage.action?pageId=155282952)
    try:
        s = smtplib.SMTP(host=smtp_server)
        s.sendmail(sender, recipients, msg.as_string())
        s.quit()
        print("Successfully sent email!")
    except smtplib.SMTPException as exc:
        print("Error sending email!")
        print(exc)


def create_plot(df: pd.DataFrame, key: str) -> None:
    def create_axhline(
        values, lowerbound, upperbound, text: str, color: str = "green"
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
    plt.xlim(1, 16)
    plt.ylim(0, 8)

    create_axhline(y, 0, 16, text="Average Score: ", color="black")
    create_axhline(y, 0, 3, text="P-Score: ")
    create_axhline(y, 3, 7, text="E-Score: ")
    create_axhline(y, 7, 10, text="R-Score: ")
    create_axhline(y, 10, 14, text="M-Score: ")
    create_axhline(y, 14, 16, text="A-Score: ")

    plt.title(f"Your overall Team PERMA score on {key.split('_')[2]}")
    plt.xlabel("PERMA Question")
    plt.ylabel("PERMA Score")
    plt.grid(True)
    plt.show()

    fig.savefig(IMG_DIR / f"{key}.png")


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

    # Compute Team PERMA, create plot and send email to team members
    for date in dates:
        for g_id in group_ids:
            key = f"group_{g_id}_{date}"

            df_avg = compute_team_perma(
                df=df[(df["Group_IDs"] == g_id) & (df["Date"] == date)],
                aggregate_func=aggregate_func,
            )

            create_plot(df_avg, key)

            if send:
                send_mail(
                    image_file=IMG_DIR / f"{key}.png",
                    recipients=get_email_recipients(df, g_id),
                )


if __name__ == "__main__":
    main()
