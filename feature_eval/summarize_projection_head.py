import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize projection head ablation linear-eval results.")
    parser.add_argument("--with-head", required=True,
                        help="JSON result for the run with projection head.")
    parser.add_argument("--without-head", required=True,
                        help="JSON result for the run without projection head.")
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def fmt(value):
    return "{:.2f}".format(value)


def main():
    args = parse_args()
    with_head = load_json(args.with_head)
    without_head = load_json(args.without_head)

    best_delta = with_head["best_test_acc"] - without_head["best_test_acc"]
    final_delta = with_head["final_test_acc"] - without_head["final_test_acc"]

    print("| Setting | Feature dim | Best acc (%) | Best epoch | Final acc (%) |")
    print("|---|---:|---:|---:|---:|")
    print("| With projection head | {} | {} | {} | {} |".format(
        with_head["feature_dim"],
        fmt(with_head["best_test_acc"]),
        with_head["best_epoch"],
        fmt(with_head["final_test_acc"])))
    print("| Without projection head | {} | {} | {} | {} |".format(
        without_head["feature_dim"],
        fmt(without_head["best_test_acc"]),
        without_head["best_epoch"],
        fmt(without_head["final_test_acc"])))
    print("")
    print("Best accuracy delta, with - without: {} percentage points".format(
        fmt(best_delta)))
    print("Final accuracy delta, with - without: {} percentage points".format(
        fmt(final_delta)))

    if best_delta > 0:
        print("Conclusion: the nonlinear projection head improves the frozen encoder representation in this setting.")
    elif best_delta < 0:
        print("Conclusion: the run without projection head gives higher linear-eval accuracy in this setting.")
    else:
        print("Conclusion: both settings give the same best linear-eval accuracy in this setting.")


if __name__ == "__main__":
    main()
