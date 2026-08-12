"""Making sense of what the pipeline produced — diagnosis of results and of the data, not scoring.

`celltrack.eval` answers "what does this configuration score". These modules answer "why", and they are a
different kind of thing: they consume a run rather than rank one, they report distributions and cuts rather
than a number, and none of them is ever in the path of a submission. Keeping them here stops a diagnostic
being mistaken for a metric, and keeps the eval package to the harness that actually chooses configurations.
"""
