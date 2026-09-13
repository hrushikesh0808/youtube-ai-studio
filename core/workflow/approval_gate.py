"""
Approval Gate

Human-in-the-loop checkpoint used at every stage that requires manual
review before the pipeline is allowed to continue (per the automation
policy: story, character designs, keyframes, test animation, voice/audio,
final video). Built once here and reused at every gate.
"""

from core.logging.logger import logger


class ApprovalGate:
    """
    Presents generated content to the user and blocks until they approve,
    request regeneration, edit their input, or quit. Nothing downstream
    should treat content as final until this returns "approve".
    """

    VALID_CHOICES = {
        "a": "approve",
        "approve": "approve",
        "r": "regenerate",
        "regenerate": "regenerate",
        "e": "edit",
        "edit": "edit",
        "q": "quit",
        "quit": "quit",
    }

    def __init__(self, stage_name: str):
        self.stage_name = stage_name

    def review(self, content: str) -> str:
        """
        Show content and prompt for a decision.

        Returns one of: "approve", "regenerate", "edit", "quit"
        """
        print("\n" + "=" * 60)
        print(f"APPROVAL REQUIRED: {self.stage_name}")
        print("=" * 60)
        print(content)
        print("=" * 60)

        while True:
            raw = input(
                "\n[A]pprove  [R]egenerate  [E]dit input  [Q]uit\n> "
            ).strip().lower()

            decision = self.VALID_CHOICES.get(raw)

            if decision is None:
                print("Invalid input. Please enter A, R, E, or Q.")
                continue

            logger.info(
                "Approval gate '%s': %s", self.stage_name, decision.upper()
            )
            return decision
