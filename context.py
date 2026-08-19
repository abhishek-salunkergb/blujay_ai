import os


KNOWLEDGE_FOLDER = "knowledge"


def load_knowledge():

    knowledge = ""

    for filename in os.listdir(KNOWLEDGE_FOLDER):

        if filename.endswith(".txt"):

            path = os.path.join(
                KNOWLEDGE_FOLDER,
                filename
            )

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as file:

                knowledge += (
                    f"\n--- {filename} ---\n"
                )

                knowledge += file.read()

    return knowledge