from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


SIZE_CONFIGS = {
    "small": {"startups": 12, "investors": 24},
    "medium": {"startups": 60, "investors": 120},
    "large": {"startups": 240, "investors": 480},
}

INDUSTRY_PROFILES = [
    {
        "industry": "AI",
        "startup_keywords": ["copilot", "automation", "workflow intelligence"],
        "investor_keywords": ["AI infrastructure", "applied ML", "workflow software"],
    },
    {
        "industry": "Fintech",
        "startup_keywords": ["payments", "treasury", "embedded finance"],
        "investor_keywords": ["fintech platforms", "payments", "financial software"],
    },
    {
        "industry": "Climate",
        "startup_keywords": ["carbon analytics", "energy optimization", "industrial decarbonization"],
        "investor_keywords": ["climate software", "decarbonization", "energy analytics"],
    },
    {
        "industry": "Healthtech",
        "startup_keywords": ["clinical operations", "care navigation", "provider workflow"],
        "investor_keywords": ["digital health", "clinical workflow", "care delivery"],
    },
    {
        "industry": "SaaS",
        "startup_keywords": ["B2B platform", "team productivity", "enterprise workflow"],
        "investor_keywords": ["B2B SaaS", "enterprise software", "workflow tools"],
    },
]

STAGES = ["Pre-Seed", "Seed", "Series A", "Series B"]
STAGE_TO_INVESTOR_PREFERENCES = {
    "Pre-Seed": ["Pre-Seed", "Seed"],
    "Seed": ["Pre-Seed", "Seed"],
    "Series A": ["Seed", "Series A"],
    "Series B": ["Series A", "Series B"],
}

GEOS = [
    ("US", "North America", "USD"),
    ("Canada", "North America", "USD"),
    ("Germany", "Europe", "EUR"),
    ("France", "Europe", "EUR"),
    ("Singapore", "Asia", "USD"),
    ("Japan", "Asia", "USD"),
]

INVESTOR_TYPES = ["VC", "Angel", "CVC"]

STAGE_RAISE_RANGES = {
    "Pre-Seed": (150_000, 750_000),
    "Seed": (500_000, 2_000_000),
    "Series A": (2_000_000, 7_000_000),
    "Series B": (7_000_000, 20_000_000),
}


def generate_dataset(
    output_dir: str | Path,
    size: str,
    overwrite: bool = False,
    seed: int | None = None,
) -> dict[str, object]:
    """Generate compatible synthetic startup, investor, and ground-truth CSVs."""

    if size not in SIZE_CONFIGS:
        raise ValueError(f"Unsupported size '{size}'. Expected one of: {', '.join(SIZE_CONFIGS)}")

    rng = random.Random(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    startups_path = output / "startups.csv"
    investors_path = output / "investors.csv"
    ground_truth_path = output / "ground_truth.csv"
    output_paths = [startups_path, investors_path, ground_truth_path]

    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not overwrite:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            f"Refusing to overwrite existing generated data without --overwrite: {existing}"
        )

    if overwrite:
        for path in output_paths:
            if path.exists():
                path.unlink()

    config = SIZE_CONFIGS[size]
    startups = _generate_startups(config["startups"], rng)
    investors, ground_truth = _generate_investors_and_ground_truth(
        startups=startups,
        investor_count=config["investors"],
        rng=rng,
    )

    _write_csv(
        startups_path,
        [
            "startup_id",
            "name",
            "description",
            "industries",
            "stage",
            "country",
            "region",
            "fundraising_amount",
            "currency",
            "website",
        ],
        startups,
    )
    _write_csv(
        investors_path,
        [
            "investor_id",
            "name",
            "description",
            "industries",
            "preferred_stages",
            "countries",
            "regions",
            "ticket_min",
            "ticket_max",
            "currency",
            "investor_type",
            "website",
        ],
        investors,
    )
    _write_csv(
        ground_truth_path,
        ["startup_id", "investor_id"],
        ground_truth,
    )

    return {
        "startup_count": len(startups),
        "investor_count": len(investors),
        "ground_truth_count": len(ground_truth),
        "startups_path": startups_path,
        "investors_path": investors_path,
        "ground_truth_path": ground_truth_path,
    }


def _generate_startups(count: int, rng: random.Random) -> list[dict[str, object]]:
    startups: list[dict[str, object]] = []
    for index in range(1, count + 1):
        primary = rng.choice(INDUSTRY_PROFILES)
        secondary = rng.choice([profile for profile in INDUSTRY_PROFILES if profile != primary])
        stage = rng.choice(STAGES)
        country, region, currency = rng.choice(GEOS)
        raise_min, raise_max = STAGE_RAISE_RANGES[stage]
        fundraising_amount = rng.randrange(raise_min, raise_max + 1, 50_000)
        industries = [primary["industry"]]
        if rng.random() < 0.35:
            industries.append(secondary["industry"])

        startups.append(
            {
                "startup_id": f"s{index}",
                "name": f"{primary['industry']} Startup {index}",
                "description": (
                    f"{primary['industry']} platform for {rng.choice(primary['startup_keywords'])} "
                    f"at the {stage} stage in {region}."
                ),
                "industries": ", ".join(industries),
                "stage": stage,
                "country": country,
                "region": region,
                "fundraising_amount": fundraising_amount,
                "currency": currency,
                "website": f"https://startup-{index}.example",
            }
        )
    return startups


def _generate_investors_and_ground_truth(
    startups: list[dict[str, object]],
    investor_count: int,
    rng: random.Random,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    investors: list[dict[str, object]] = []
    ground_truth: list[dict[str, str]] = []

    for index, startup in enumerate(startups, start=1):
        investor_id = f"i{index}"
        primary_industry = startup["industries"].split(",")[0].strip()
        profile = next(item for item in INDUSTRY_PROFILES if item["industry"] == primary_industry)
        stage = str(startup["stage"])
        country = str(startup["country"])
        region = str(startup["region"])
        currency = str(startup["currency"])
        raise_amount = int(startup["fundraising_amount"])

        investors.append(
            {
                "investor_id": investor_id,
                "name": f"{primary_industry} Ventures {index}",
                "description": (
                    f"Fund focused on {', '.join(STAGE_TO_INVESTOR_PREFERENCES[stage])} companies in "
                    f"{rng.choice(profile['investor_keywords'])} across {region}."
                ),
                "industries": startup["industries"],
                "preferred_stages": ", ".join(STAGE_TO_INVESTOR_PREFERENCES[stage]),
                "countries": country,
                "regions": region,
                "ticket_min": max(100_000, int(raise_amount * 0.4)),
                "ticket_max": int(raise_amount * 1.8),
                "currency": currency,
                "investor_type": rng.choice(INVESTOR_TYPES),
                "website": f"https://investor-{index}.example",
            }
        )
        ground_truth.append({"startup_id": str(startup["startup_id"]), "investor_id": investor_id})

    for index in range(len(startups) + 1, investor_count + 1):
        profile = rng.choice(INDUSTRY_PROFILES)
        stage = rng.choice(STAGES)
        country, region, currency = rng.choice(GEOS)
        raise_min, raise_max = STAGE_RAISE_RANGES[stage]
        midpoint = (raise_min + raise_max) // 2
        investors.append(
            {
                "investor_id": f"i{index}",
                "name": f"{profile['industry']} Capital {index}",
                "description": (
                    f"Investor backing {rng.choice(profile['investor_keywords'])} "
                    f"with a focus on {stage} and later across {region}."
                ),
                "industries": profile["industry"],
                "preferred_stages": stage,
                "countries": country,
                "regions": region,
                "ticket_min": max(100_000, int(midpoint * 0.7)),
                "ticket_max": int(midpoint * 2.5),
                "currency": currency,
                "investor_type": rng.choice(INVESTOR_TYPES),
                "website": f"https://investor-{index}.example",
            }
        )

    return investors, ground_truth


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic startup/investor benchmark data.")
    parser.add_argument("--size", choices=sorted(SIZE_CONFIGS), default="small")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    summary = generate_dataset(
        output_dir=Path("data") / "generated",
        size=args.size,
        overwrite=args.overwrite,
        seed=args.seed,
    )
    print(
        "\n".join(
            [
                f"Generated startups: {summary['startup_count']}",
                f"Generated investors: {summary['investor_count']}",
                f"Generated ground truth matches: {summary['ground_truth_count']}",
                f"Startups path: {summary['startups_path']}",
                f"Investors path: {summary['investors_path']}",
                f"Ground truth path: {summary['ground_truth_path']}",
            ]
        )
    )


if __name__ == "__main__":
    main()
