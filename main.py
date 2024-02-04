import streamlit as st
from langchain.schema import (
    HumanMessage,
    AIMessage
)
from agent import get_response


def main() -> None:

    st.header("AI Math Tutor")

    # Display chat messages from history
    messages = st.session_state.get('messages', [])
    # for message in messages:
    #     st.chat_message(message["role"]).markdown(message["content"])
    for i, msg in enumerate(messages):
        if i % 2 == 0:
            st.chat_message("user").write(msg.content)
        else:
            st.chat_message("assistant").write(msg.content)


    # Only execute on the first run
    if "messages" not in st.session_state:
        st.session_state.messages = []
        

    # Next conversation - user input and assistant response
    if prompt := st.chat_input("Your message: "):

        # Display user message and append to session state messages
        st.chat_message("user").write(prompt)
        st.session_state.messages.append(
            # {'role':'user', 'content': prompt}
            HumanMessage(content=prompt)
        )

        # Get AI response based on the user input
        generate_response(prompt, st.session_state.messages)

    return


def generate_response(prompt: str, messages: list) -> None:
    """
    Generate llm response from messages and then use the response to generate DALL-E image.
    Display the AI generated response and image in the conversation window.
    """

    with st.chat_message("assistant"):

        with st.spinner("AI is thinking ..."):
            response = get_response(prompt)

        messages.append(
        #{"role": "assistant", "content": full_response}
            AIMessage(content=response)
        )

        st.write(response)

    return



if __name__ == "__main__":
    main()