"""
Verify identity-disjointness directly from the OFFICIAL files, bypassing
the manifest entirely. This tells us if the issue is (a) our manifest build,
or (b) an inherent property of the official Celeb-DF split.

It reads:
  - List_of_testing_videos.txt  (defines test videos)
  - the three folders (everything not in the list = train)
and computes TARGET-identity overlap straight from filenames.

Run:
  python verify_official_split.py --root ../Dataset/Celeb-DF-v2
"""
import argparse, os, re, glob


def target_id(path, folder):
    base = os.path.splitext(os.path.basename(path))[0]
    ids = re.findall(r"id(\d+)", base)
    if "synthesis" in folder.lower():
        return f"id{ids[0]}" if ids else None      # target = first id
    if "real" in folder.lower() and ids:
        return f"id{ids[0]}"
    return None  # YouTube-real numeric -> no celeb identity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    test_list = os.path.join(args.root, "List_of_testing_videos.txt")
    test_rel = set()
    with open(test_list) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            test_rel.add(line.split()[1].replace("\\", "/"))
    print(f"[info] official test list entries: {len(test_rel)}")

    folders = ["Celeb-real", "YouTube-real", "Celeb-synthesis"]
    train_t, test_t = set(), set()
    train_n = test_n = 0
    for folder in folders:
        for vp in glob.glob(os.path.join(args.root, folder, "*.mp4")):
            rel = f"{folder}/{os.path.basename(vp)}"
            tid = target_id(vp, folder)
            if rel in test_rel:
                test_n += 1
                if tid: test_t.add(tid)
            else:
                train_n += 1
                if tid: train_t.add(tid)

    print(f"[info] train videos={train_n} test videos={test_n}")
    print(f"[info] train target-ids={len(train_t)} test target-ids={len(test_t)}")
    overlap = train_t & test_t
    print(f"\n*** TARGET-identity overlap (official split): {len(overlap)} ***")
    if overlap:
        print("    examples:", sorted(overlap)[:15])
        print("\n  => The OFFICIAL split itself shares target identities.")
        print("     This is NOT a manifest-build error; it is how the official")
        print("     List_of_testing_videos.txt is defined (Celeb-DF has ~59 subjects).")
    else:
        print("  => Official split IS target-identity disjoint. If your manifest")
        print("     showed overlap, the manifest build is the problem -> fixable.")


if __name__ == "__main__":
    main()
