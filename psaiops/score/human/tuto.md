The app is essentially asking the LLM "would you have generated the same response?".
Then it probes the internals of the model for its opinion.
The more surprised the LLM is the more "human" the text is scored.

Some text that were originally human written are considered "AI" here:
- very well-know samples like Wikipedia articles, or classic books in the public domain
- frozen and predictable phrases like salutations

Anything that is highly predictable is deemed "AI": more precisely, this app asks the question "what is *original* human work in this text?".

The final probability score is a combination of several metrics computed token by token:
- the `unicode` metric outlines rare glyphs like the famous "em-dash" that is not typically available on human keyboards
- the `surprisal` metric measures how individual tokens fit into the distribution estimated by the model
- the `perplexity` metric rates how a sequence of tokens fits together
- the `ramping` ratio is used to dampen the scores at the begining, where the model lacks context

The fixed recipe used to combine these metrics into the final score for each token is obviously insufficient to detect the ever improving outputs of LLMs.

It is highly likely you make a better use of them, so the internal of the LLM are plotted in the "graphs" tab.
You will find a few samples to train your eye and spot LLM patterns.

And the details of the scoring recipe are available in the tab "docs".
