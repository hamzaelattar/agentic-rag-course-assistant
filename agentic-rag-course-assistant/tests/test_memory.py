
import sys

from app.agent.memory import (
    open_conversation_agent,
    ask_agent,
    get_conversation_history
)


def main():

    thread_id = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "conversation_001"
    )

    print("\n===== AGENTIC RAG AVEC MÉMOIRE =====")
    print("Conversation :", thread_id)

    with open_conversation_agent() as agent:

        history = get_conversation_history(
            agent,
            thread_id
        )

        print(
            "Messages précédents :",
            len(history)
        )

        for message in history:
            print(
                f"{message['role']} : "
                f"{message['content'][:250]}"
            )

        print("\nTape 'exit' pour quitter.")

        while True:

            question = input(
                "\nÉtudiant : "
            ).strip()

            if question.lower() in {
                "exit", "quit", "quitter"
            }:
                print("Conversation enregistrée.")
                break

            if not question:
                continue

            result = ask_agent(
                agent,
                thread_id,
                question
            )

            print("\nAssistant :")
            print(result.get("answer", ""))

            print(
                "\nOutils :",
                result.get("selected_tools", [])
            )

            print(
                "Question reformulée :",
                result.get("rewritten_question", "")
            )

            history = get_conversation_history(
                agent,
                thread_id
            )

            print(
                "Messages en mémoire :",
                len(history)
            )


if __name__ == "__main__":
    main()
