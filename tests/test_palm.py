import warnings
import os

with warnings.catch_warnings():
    # google cloud uses deprecated apis
    # suppress the warning locally
    warnings.simplefilter("ignore")
    import google.generativeai as palm  # type: ignore


palm.configure(api_key=os.environ.get("PALM_API_KEY"))


def test_palm_completion():
    prompt = """
    You are an expert at solving word problems.

    Solve the following problem:

    I have three houses, each with three cats.
    each cat owns 4 mittens, and a hat. Each mitten was
    knit from 7m of yarn, each hat from 4m.
    How much yarn was needed to make all the items?

    Think about it step by step, and show your work.
    """
    completion = palm.generate_text(
        model="models/text-bison-001",
        prompt=prompt,
        temperature=0,
        # The maximum length of the response
        max_output_tokens=800,
    )

    print(completion.result)
