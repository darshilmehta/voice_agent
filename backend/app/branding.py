"""The product's name and tagline (decided 2026-10-10, docs/DESIGN.md).

Docent is what users see and what the assistant calls itself. Internal identifiers (the repo ``voice_agent``, the folder
``poc_gibberlink``, package, container, collection and config-key names) keep their old names on purpose.
"""

PRODUCT_NAME = "Docent"
TAGLINE = "Talk to your documents"
TAGLINE_HI = "अपने दस्तावेज़ों से बात करें"

# One line in every prompt where the assistant speaks as itself: it answers "who are you?" with its name, in the
# language the user is speaking (the name stays in Latin script; "Docent" is read the same way in Hindi).
IDENTITY = (
    f"Your name is {PRODUCT_NAME}. If asked who or what you are, say you are {PRODUCT_NAME}, an assistant that "
    "talks with the user about their documents, out loud or in text, in English and Hindi. Never call yourself "
    "anything else (not a language model, not a bot from any company)."
)
