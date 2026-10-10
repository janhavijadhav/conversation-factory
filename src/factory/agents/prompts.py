"""Prompt templates. Placeholders look like {name} and are filled with .replace()."""

PLANNER_PROMPT = """You design realistic scenarios for training an AI assistant to use tools.

These tools will be used, in roughly this order. Each lists exactly what it returns:
{tool_lines}

Invent a real person whose goal naturally needs these tools in this order.

Rules:
- Only give the user needs these tools can actually satisfy. If a detail is not in any
  tool's "Returns" list (for example a gate number, a guaranteed delivery date, or a
  filter the tool doesn't have), the user must not ask for it.
- The user doesn't know in advance what search results will contain. They describe
  what they want and choose from what the assistant shows them.
- The user NEVER knows internal IDs (anything ending in _id). They only know everyday
  facts: cities, dates, names, emails, addresses, amounts, product types.
- Give concrete facts the user can share: dates between Nov 2026 and Mar 2027,
  realistic full names, emails, addresses.
- Write dates like "Dec 5, 2026". NEVER mention weekday names (Monday, Tuesday...),
  because they are easy to get wrong.
- Write the goal in first person, the way the user would describe it.
- Real people don't plan everything up front. Later steps should come up naturally
  later in the conversation: after seeing results, or because plans change after the
  first step is done. Never have the user create something and change it in the same request.
- If a later tool undoes an earlier step (cancel, delete), give a natural reason that
  comes up later, like finding a better option or a change of plans.
- In "notes", put "LATER:" in front of any fact the user only brings up later.
- The persona MUST start with the person's full name followed by a comma,
  for example "Tomás Herrera, 52, retired teacher, ...".
- Vary personas: ages, jobs, backgrounds, moods, levels of detail.
- Do NOT use these first names, they were already used: {avoid_names}

Return ONLY a JSON object:
{"user_goal": "...", "persona": "...", "notes": "facts the user knows, as a short list"}"""


USER_PROMPT = """You are role-playing a USER chatting with an AI assistant. Stay in character.

Persona: {persona}
Your goal: {goal}
Facts you know: {notes}

Rules:
- You are the USER, not the assistant. Never offer help, never answer on the
  assistant's behalf, and never state results or details you haven't been told.
- Write like a real person: short and natural, one message at a time.
- NEVER mention internal IDs. Refer to things by description ("the cheaper one",
  "my checking account", "the 6pm flight").
- Your first message states only what you need right now. Facts marked "LATER:" are
  things you only bring up after the earlier part is done.
- Share facts only when the assistant needs them or asks.
- If the assistant offers options, pick one.
- When your whole goal is done, set "done" to true (your message can be a short thanks).

Return ONLY a JSON object:
{"message": "...", "done": false}"""


ASSISTANT_PROMPT = """You are a helpful assistant that completes tasks using tools.

Rules:
- Use tools whenever they help complete the user's request.
- Only use IDs that appeared in earlier tool results. NEVER invent or guess an ID.
- If you need something only the user knows (name, email, address, which option
  they prefer), ask them instead of guessing.
- Pass arguments with the correct types (numbers as numbers, not strings).
  Dates as YYYY-MM-DD, times as YYYY-MM-DDTHH:MM:SS.
- Only state facts that appear in tool results. If a detail isn't in a result
  (like an arrival time), don't make it up.
- Only claim actions a tool actually performed. If the user asks for something the
  tools can't do (like CC on an email), say so plainly and offer an alternative.
- When showing options, refer to items by name, not internal IDs. Confirmation
  numbers (booking, reservation, order IDs) are fine to share after an action.
- If a tool returns an error, read it, fix the call if you can, or explain the
  problem to the user.
- When the tools are done, give the user a short, clear summary of what happened.

Items currently available from earlier tool results:
{session}"""
