Code to supplement Karjus 2023, "Machine-assisted mixed methods: augmenting humanities and social sciences with artificial intelligence".

Preprint: https://arxiv.org/abs/2309.14379 <br>
Twitter/X explainer thread: https://twitter.com/AndresKarjus/status/1706959842227659112

Comments, critique, feedback and ideas for improvements are all warmly welcome: see [andreskarjus.github.io](https://andreskarjus.github.io) for contacts.

This code is provided as-is, with no implication of warranty or liability. This particular implementation relies on using the OpenAI API, which is subject to possible changes. The API usage requires setting up an OpenAI account and within that, a billing account. The code and prompts here can of course be adapted to be used on a local LLM if desired.

The code is in R and assumes at least R 4.2.1. Open the case_studies file in R (or RStudio) for instructions. The required packages are in the requirements.R file. Sourcing it will install them, as well as the reticulate package which is used for Python parts of the code, including the OpenAI packages (which are not available in R). The already computed results are also made available (see case studies file for instructions).
 
