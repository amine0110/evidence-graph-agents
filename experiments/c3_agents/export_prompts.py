"""Export every prompt the paper reports into frozen/prompts.json.

Current prompts are read from the code that ran them. The two replaced prompts
(caption-v1, speech-tag-v1) are kept here verbatim: caption-v1 is the prompt of
commit b587c23; speech-tag-v1 ran before the speech tagger was first committed,
and its outputs carry the version string "speech-tag-v1".
"""

import json
from pathlib import Path

import reasoning
import speech_tag_pass
import task
import video_pass

CAPTION_V1 = ("These are {n} frames, one second apart, from a video. In at most 40 words, "
              "describe factually what the people do and what happens. Mention weapons, fire, "
              "vehicles or injuries only if you see them.")
SPEECH_TAG_V1_LABELS = ("A: threat or aggression: the speaker threatens someone, gives orders at weapon point, "
                        "or verbally attacks someone.\n"
                        "B: distress: the speaker screams, begs, calls for help, or expresses pain or fear.")


def main():
    v2_labels = "\n".join(l for l in speech_tag_pass.PROMPT.splitlines() if l.startswith(("A:", "B:")))
    prompts = {
        "task_question": task.QUESTION,
        "s3_system": reasoning.S3_SYSTEM % reasoning.MAX_STEPS,
        "s3_paraphrased_opening": reasoning.S3_INTRO_P2,
        "s3v2_system": reasoning.S3V2_SYSTEM.replace("{intro}", reasoning.S3V2_INTRO["c"]) % reasoning.MAX_STEPS_V2,
        "s4_system": reasoning.S4_SYSTEM,
        "s4_paraphrased_system": reasoning.S4_SYSTEM_P2,
        "s2_agent_roles": reasoning.AGENT_ROLE,
        "s2_report_request": reasoning.REPORT_ASK,
        "s2_reasoning_system": reasoning.S2_SYSTEM % reasoning.MAX_QUESTIONS,
        "cite_line": reasoning.CITE_LINE,
        "speech_tag": speech_tag_pass.PROMPT,
        "speech_tag_v1_labels": SPEECH_TAG_V1_LABELS,
        "speech_tag_v2_labels": v2_labels,
        "caption_v1": CAPTION_V1,
        "caption_v2": video_pass.PROMPT,
    }
    out = Path(__file__).parent / "frozen" / "prompts.json"
    out.write_text(json.dumps(prompts, indent=1, ensure_ascii=False) + "\n")
    print(sorted(prompts))


if __name__ == "__main__":
    main()
