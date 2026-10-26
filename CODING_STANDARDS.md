# Coding Standards

## Code Style

- **No emojis**: Do not use emojis in code or commit messages.
- **Minimal comments**: Comment only when behavior is not self-explanatory, such as complex algorithms or non-obvious optimizations. Comments explain *why*, never the history of the change.

### The reader test

Assume your reader has the repository as it stands and nothing else. No chat history, no earlier version of the file, no memory of the discussion behind the change. Everything they need in order to understand the code must be in the code.

### Integration

Before implementing a feature or adding a constant, helper, or type, search for one that already exists. Same concept, one definition.

When something close to what you need already exists, change it instead of copying it. Updating its callers is part of your change. Two functions that differ by one argument should be one function that takes that argument. But do not couple two things that merely look alike.

If you change code that other places call, check that every caller still behaves the same. That is the author's job, and a reviewer can ask which callers you checked.

### Scope

State what the pull request does, then write the smallest diff that does all of it, including the integration work the change needs to actually be used. Small is not the same as partial, so do not leave half the change for a follow-up.

- Solve the proposed problem only.
- Leave unrelated `TODO`s and `FIXME`s alone, even when editing the same file.
- Keep diffs focused. Land unrelated reformatting and renames separately.
- Extend the local idiom rather than converting surrounding code to a new one.
- No scratch files, logs, or one-off scripts unless they are the point of the change.
- A behavior change comes with a test that fails without it.
- Don't hardcode the credentials in script

### Simplicity

Write the minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

### Naming and structure

- Names state intent and units, written out in full: `ev_bb` over `ev`, `reach_prob` over `rp`.
- Use the domain vocabulary already in the codebase.
- Prefer a named function or variable over a comment introducing a block.
- Match the surrounding file's conventions.

## Writing Style

Applies to comments, commit messages, pull request descriptions, and docs.

- Be objective. Replace adjectives and adverbs with data. Instead of
  "much faster," write "reduced p90 latency from 10 ms to 1 ms."
- No weasel words: "significantly," "arguably," "nearly all," "should
  result in," "might help." If you can't back a claim with a number, state
  the claim plainly or drop it.
- Keep sentences under 30 words. Prefer "because" over "due to the fact
  that," "could not" over "totally lacked the ability to."
- No project-specific jargon or acronyms that exclude newcomers. Spell them
  out on first use, e.g. "Non-Disclosure Agreement (NDA)." Standard
  engineering terms (`TODO`, p90, API) are fine.
- Every sentence must pass the "so what" test. Cut anything that doesn't
  change the reader's understanding or decision.
