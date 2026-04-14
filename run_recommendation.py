import argparse
import ast
import asyncio
import json

from plant_data.router import RecommendationInput, recommend_plants


YES_VALUES = {"y", "yes", "true", "1"}
NO_VALUES = {"n", "no", "false", "0", ""}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run plant recommendations locally without Postman."
    )
    parser.add_argument(
        "--input-json",
        help="Recommendation input as JSON or a Python-style dict string.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for recommendation inputs interactively.",
    )
    return parser.parse_args()


def default_payload() -> dict:
    return {
        "budget": 8,
        "max_recommendations": 5,
        "min_popularity_rating": 1,
        "flood_tolerant": True,
        "sun_full": True,
        "moisture_wet": True,
        "pos_base": True,
        "benefits_pollinators": True,
    }


def parse_input_payload(raw_value: str) -> dict:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "Could not parse --input-json. Use valid JSON or run with --interactive."
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError("--input-json must parse to an object/dictionary.")
        return parsed


def ask_text(prompt: str) -> str:
    return input(f"{prompt}: ").strip()


def ask_int(prompt: str, default: int | None = None) -> int:
    while True:
        label = f"{prompt} (press Enter for {default})" if default is not None else prompt
        value = ask_text(label)
        if value == "" and default is not None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            print("Please enter a whole number.")


def ask_float(prompt: str, default: float | None = None) -> float:
    while True:
        label = f"{prompt} (press Enter for {default})" if default is not None else prompt
        value = ask_text(label)
        if value == "" and default is not None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            print("Please enter a number.")


def ask_optional_float(prompt: str) -> float | None:
    while True:
        value = ask_text(f"{prompt} (leave blank to skip)")
        if value == "":
            return None
        try:
            return float(value)
        except ValueError:
            print("Please enter a number or leave it blank.")


def ask_optional_text(prompt: str) -> str | None:
    value = ask_text(f"{prompt} (leave blank to skip)")
    return value or None


def ask_bool(prompt: str, default: bool = False) -> bool:
    while True:
        value = input(f"{prompt} [y/n]: ").strip().lower()
        if value == "":
            return default
        if value in YES_VALUES:
            return True
        if value in NO_VALUES:
            return False
        print("Please enter y or n.")


def print_intro() -> None:
    print("Rain Garden Recommendation Tester")
    print("Answer the questions below to generate plant recommendations.")
    print("Press Enter to use the suggested value when one is shown.")


def print_section(title: str, description: str) -> None:
    print(f"\n[{title}]")
    print(description)


def format_score_breakdown(score_breakdown: dict[str, float]) -> str:
    labels = {
        "price": "Price",
        "popularity": "Popularity",
        "tolerances": "Tolerances",
        "sun": "Sun",
        "moisture": "Moisture",
        "position": "Position",
        "space": "Space",
        "soil": "Soil",
        "benefits": "Benefits",
    }
    parts = []
    for key, value in score_breakdown.items():
        parts.append(f"{labels.get(key, key)} {value:.2f}")
    return " | ".join(parts)


def build_interactive_payload() -> dict:
    print_intro()

    print_section(
        "Ranking Controls",
        "These set budget, popularity filtering, and how many recommendations you want to see.",
    )
    payload = {
        "budget": ask_float(
            "Budget score for plant cost fit. Lower favors cheaper plants. Suggested range 3 to 10",
            8,
        ),
        "max_recommendations": ask_int(
            "How many recommendations would you like",
            5,
        ),
        "min_popularity_rating": ask_int(
            "Minimum popularity rating to allow. Use 0 to keep everything or 1 to 5 to filter",
            1,
        ),
    }

    print_section(
        "Sun Exposure",
        "Select every light condition your planting area can handle.",
    )
    payload.update(
        {
            "sun_full": ask_bool("Full sun: 6 or more hours of direct sun"),
            "sun_partial": ask_bool("Partial sun: mixed sun and shade"),
            "sun_shade": ask_bool("Shade: mostly shaded"),
        }
    )

    print_section(
        "Moisture",
        "Choose the actual moisture pattern of the planting area.",
    )
    payload.update(
        {
            "moisture_wet": ask_bool("Wet soil: often saturated or slow-draining"),
            "moisture_med": ask_bool("Medium moisture: normal garden moisture"),
            "moisture_dry": ask_bool("Dry soil: drains fast or dries out"),
        }
    )

    print_section(
        "Rain Garden Position",
        "Choose where the plant will sit within the rain garden.",
    )
    payload.update(
        {
            "pos_base": ask_bool("Base: lowest basin area"),
            "pos_slope": ask_bool("Slope: side wall or transition zone"),
            "pos_margin": ask_bool("Margin: upper edge or outer border"),
        }
    )

    print_section(
        "Stress Tolerances",
        "Mark these only when the plant must handle these site stresses.",
    )
    payload.update(
        {
            "flood_tolerant": ask_bool("Must tolerate flooding"),
            "drought_tolerant": ask_bool("Must tolerate drought"),
            "road_salt_tolerant": ask_bool("Must tolerate road salt"),
        }
    )

    print_section(
        "Wildlife Benefits",
        "Choose whether wildlife support matters for this recommendation.",
    )
    wildlife_support = ask_bool("Should the plant support wildlife like pollinators, birds, butterflies, or hummingbirds")
    payload.update(
        {
            "benefits_pollinators": wildlife_support,
            "benefits_butterflies": wildlife_support,
            "benefits_hummingbirds": wildlife_support,
            "benefits_birds": wildlife_support,
        }
    )

    print_section(
        "Plant Size / Spacing",
        "Optional. Spacing means how much width a mature plant usually needs between neighboring plants.",
        )
    print("Examples:")
    print("- 12 to 18 inches for compact plants")
    print("- 18 to 36 inches for medium plants")
    print("- Leave blank if spacing does not matter")
    payload.update(
        {
            "space_min": ask_optional_float("Smallest spacing you want in inches"),
            "space_max": ask_optional_float("Largest spacing you want in inches"),
        }
    )

    print_section(
        "Soil Preferences",
        "Optional. Use this only if you want to match soil conditions at the planting site.",
    )
    print("Soil type examples: clay, loam, sandy")
    print("Soil pH examples: acidic, neutral, alkaline")
    print("Leave blank if soil should not affect results")
    payload.update(
        {
            "soil_pref": ask_optional_text("Preferred soil type"),
            "soil_ph": ask_optional_text("Preferred soil pH"),
        }
    )

    return payload


def print_recommendation_summary(payload: dict, result) -> None:
    print("\nYour Inputs")
    print(f"Budget: {payload.get('budget')}")
    print(f"Recommendations requested: {payload.get('max_recommendations')}")
    print(f"Minimum popularity: {payload.get('min_popularity_rating')}")

    active_tags = []
    for key, label in [
        ("sun_full", "Full Sun"),
        ("sun_partial", "Partial Sun"),
        ("sun_shade", "Shade"),
        ("moisture_wet", "Wet Soil"),
        ("moisture_med", "Medium Moisture"),
        ("moisture_dry", "Dry Soil"),
        ("pos_base", "Base"),
        ("pos_slope", "Slope"),
        ("pos_margin", "Margin"),
        ("flood_tolerant", "Flood Tolerant"),
        ("drought_tolerant", "Drought Tolerant"),
        ("road_salt_tolerant", "Road Salt Tolerant"),
        ("benefits_pollinators", "Pollinators"),
        ("benefits_butterflies", "Butterflies"),
        ("benefits_hummingbirds", "Hummingbirds"),
        ("benefits_birds", "Birds"),
    ]:
        if payload.get(key):
            active_tags.append(label)
    if payload.get("space_min") is not None or payload.get("space_max") is not None:
        active_tags.append(
            f"Spacing {payload.get('space_min', '?')} to {payload.get('space_max', '?')} in"
        )
    if payload.get("soil_pref"):
        active_tags.append(f"Soil {payload['soil_pref']}")
    if payload.get("soil_ph"):
        active_tags.append(f"Soil pH {payload['soil_ph']}")

    print("Preferences: " + (", ".join(active_tags) if active_tags else "None"))

    print("\nRecommendations")
    if not result.recommendations:
        print("No recommendations matched these inputs.")
        return

    for index, plant in enumerate(result.recommendations, start=1):
        print(f"\n{index}. {plant.display_name}")
        print(f"   Plant ID: {plant.plant_id}")
        print(f"   Match Score: {plant.score:.3f}")
        print(f"   Predicted Price: {plant.predicted_price:.2f}")
        print(f"   Popularity: {plant.popularity_rating}")
        print(f"   Why it matched: {', '.join(plant.matched_preferences) if plant.matched_preferences else 'General fit'}")
        print(f"   Score details: {format_score_breakdown(plant.score_breakdown)}")


async def main() -> None:
    args = parse_args()
    try:
        if args.interactive:
            payload = build_interactive_payload()
        elif args.input_json:
            payload = parse_input_payload(args.input_json)
        else:
            payload = build_interactive_payload()
    except ValueError as exc:
        print(str(exc))
        return

    request = RecommendationInput(**payload)
    result = await recommend_plants(request)
    print_recommendation_summary(payload, result)


if __name__ == "__main__":
    asyncio.run(main())
