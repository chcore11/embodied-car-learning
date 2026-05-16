import csv
from collections import defaultdict

path = r"data\datasets\v0_5\expert_demonstrations.csv"

rows = []
with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row["step"] = int(row["step"])
        row["x"] = int(row["x"])
        row["y"] = int(row["y"])
        rows.append(row)

bad_forward_blocked = [
    r for r in rows
    if r["action"] == "forward" and r["front_blocked"] in ("1", "true", "True")
]

bad_forward_true_blocked = [
    r for r in rows
    if r["action"] == "forward" and r["true_front_blocked"] in ("1", "true", "True")
]

print("Rows:", len(rows))
print("action=forward and front_blocked=1:", len(bad_forward_blocked))
print("action=forward and true_front_blocked=1:", len(bad_forward_true_blocked))

print("\nExamples of action=forward and true_front_blocked=1:")
for r in bad_forward_true_blocked[:10]:
    print({
        "case_name": r["case_name"],
        "episode_id": r["episode_id"],
        "step": r["step"],
        "x": r["x"],
        "y": r["y"],
        "direction": r["direction"],
        "front_blocked": r["front_blocked"],
        "true_front_blocked": r["true_front_blocked"],
        "action": r["action"],
        "reward": r["reward"],
    })

episodes = defaultdict(list)
for r in rows:
    episodes[r["episode_id"]].append(r)

print("\nSuspicious transitions:")
count = 0
for episode_id, ep_rows in episodes.items():
    ep_rows.sort(key=lambda r: r["step"])
    for i in range(len(ep_rows) - 1):
        cur = ep_rows[i]
        nxt = ep_rows[i + 1]

        if cur["action"] != "forward":
            continue

        moved = (cur["x"], cur["y"]) != (nxt["x"], nxt["y"])
        true_blocked = cur["true_front_blocked"] in ("1", "true", "True")

        if true_blocked and moved:
            print({
                "episode_id": episode_id,
                "case_name": cur["case_name"],
                "step": cur["step"],
                "cur": (cur["x"], cur["y"], cur["direction"]),
                "next": (nxt["x"], nxt["y"], nxt["direction"]),
                "true_front_blocked": cur["true_front_blocked"],
                "action": cur["action"],
                "reward": cur["reward"],
            })
            count += 1

print("\ntrue_front_blocked=1 but forward still moved:", count)