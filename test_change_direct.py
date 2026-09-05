import os
from dotenv import load_dotenv
load_dotenv()
from app.services import change_service

if __name__ == "__main__":
    before = "S2A_MSIL2A_20250118T053141_N0511_R105_T43QCA_20250118T092848"
    after = "S2B_MSIL2A_20250123T053029_N0511_R105_T43QCA_20250123T084836"
    print("Running...")
    res = change_service.run_change_analysis(before, after, "abs_diff", 0.15, True)
    print("Success:")
    print(res)
