# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example Notebook
#
# This is a template example notebook managed by
# [jupytext](https://jupytext.readthedocs.io).
#
# **Workflow:**
#
# - Edit this `.py` file — it is the source of truth tracked in git.
# - Run `pixi r nb-sync` to generate/update the paired `.ipynb`.
# - Run `pixi r nb-execute` to execute the notebook in-place.
# - Run `pixi r nb-run` to do both in one step (sync → execute).
# - Run `pixi r nb-clear` to strip outputs before committing.

# %%
from __future__ import annotations

import package_name

# %% [markdown]
# ## Basic Usage

# %%
result = package_name.hello("world")

# %% [markdown]
# ## Version

# %%
