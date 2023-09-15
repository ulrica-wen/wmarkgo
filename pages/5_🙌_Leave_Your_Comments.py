from datetime import datetime
from blind_watermark import db
import streamlit as st

st.set_page_config(
    page_title="Leave Your Comments",
    page_icon=":Raising Hands:",
    layout='wide'
)

st.title('🙌  Leave Your Comments')

COMMENT_TEMPLATE_MD = """{} - {}
> {}"""


# Comments part

conn = db.connect()
comments = db.collect(conn)

with st.expander("💬 Open comments"):

    # Show comments

    st.write("**Comments:**")

    for index, entry in enumerate(comments.itertuples()):
        st.markdown(COMMENT_TEMPLATE_MD.format(entry.name, entry.date, entry.comment))

        is_last = index == len(comments) - 1
        is_new = "just_posted" in st.session_state and is_last
        if is_new:
            st.success("☝️ Your comment was successfully posted.")

    st.markdown("\n")
    st.markdown('\n')
    st.markdown('\n')

    # Insert comment

    st.write("**Add your own comment:**")
    form = st.form("comment")
    name = form.text_input("Name")
    comment = form.text_area("Comment")
    submit = form.form_submit_button("Add comment")

    if submit:
        date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        db.insert(conn, [[name, comment, date]])
        if "just_posted" not in st.session_state:
            st.session_state["just_posted"] = True
        st.experimental_rerun()