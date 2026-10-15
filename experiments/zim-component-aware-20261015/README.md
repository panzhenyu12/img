# Component-aware ZIM candidate acceptance (new explicit semantics)

This experiment reuses the exact 50 holdout and 30 development raw outputs. It
never runs inference, changes prompts, modifies raw alphas, or alters production.
It keeps positive alpha >= .5, negative alpha <= .1, skin-core retained <= .05,
and hair-core retained >= .65 unchanged. The only candidate-selection change is
that hair retention is measured over the confident core of the source component
named by that prompt, instead of over every hair component in the image.

All original global-denominator selections are retained alongside the component
selections. A component with no confident reference core must abstain. Selected
raw alphas are combined by maximum only for a separately labelled audit union;
partial or abstained components are never called a complete safe matte. Global
union/core metrics, each component, low skin alpha mean/p95, and edge evidence
are reported. Parser agreement is not physical alpha accuracy.
