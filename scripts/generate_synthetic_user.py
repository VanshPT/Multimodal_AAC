import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
USER_ID = "demo_user"
OUT_DIR = BASE_DIR / "data" / "synthetic_users" / USER_ID


def utc_day(offset=0):
    return (datetime.now(timezone.utc) + timedelta(days=offset)).strftime("%Y-%m-%d")


def generate_ltm():
    friends = [
        {"person_id": "omer", "name": "Omer", "relation": "friend", "style_hint": "short playful check-ins"},
        {"person_id": "vansh", "name": "Vansh", "relation": "friend", "style_hint": "project-focused and concise"},
        {"person_id": "siddharth", "name": "Siddharth", "relation": "friend", "style_hint": "calm detailed planning"},
        {"person_id": "aditya", "name": "Aditya", "relation": "friend", "style_hint": "energetic casual tone"},
    ]
    return {
        "user_id": USER_ID,
        "display_name": "Akash",
        "age_range": "early_20s",
        "city_context": "college_town",
        "living_situation": "shares apartment with roommate",
        "primary_language": "English",
        "secondary_language": "Spanish phrases with family",
        "education_context": "CSE 635 coursework",
        "communication_style": {
            "preferred_length": "short",
            "emoji_level": "low",
            "tone": "warm_direct",
            "pace_preference": "one_idea_at_a_time",
            "repair_strategy": "clarify_then_confirm",
            "preferred_openers": ["I want to say", "Quick note", "Can we"],
            "dispreferred_openers": ["As an AI", "Technically speaking"],
            "preferred_confirmation_pattern": "repeat key plan + ask yes/no",
            "preferred_clarification_pattern": "ask one focused follow-up",
            "avoid_patterns": ["overly formal", "long explanations without pause"],
        },
        "personality_traits": [
            "kind",
            "dry humor",
            "patient",
            "values clarity",
            "likes collaborative planning",
            "likes predictable schedules",
            "prefers practical conversations",
            "enjoys small jokes with friends",
        ],
        "strengths": ["planning ahead", "task prioritization", "active listening"],
        "communication_needs": [
            "needs extra time for multi-part questions",
            "prefers one question at a time",
            "benefits from confirmation before transitions",
            "likes visual schedule reminders",
        ],
        "favorites": {
            "foods": ["veggie dumplings", "mango smoothie", "tomato soup"],
            "activities": ["movie night", "co-op games", "park walk"],
            "music": ["indie pop", "lofi"],
            "shows": ["Only Murders in the Building", "The Great British Bake Off"],
            "snacks": ["almonds", "dark chocolate", "fruit cups"],
            "drinks": ["sparkling water", "mint tea"],
            "study_spots": ["quiet corner in library", "window table at cafe"],
        },
        "relationships": {
            "omer": "Close friend, chats in short playful style.",
            "vansh": "Close friend, often discusses project deadlines.",
            "siddharth": "Close friend, helps with planning and reminders.",
            "aditya": "Close friend, casual upbeat conversation partner.",
        },
        "people_and_relationships": friends,
        "important_places": [
            "apartment building lobby",
            "campus computer lab",
            "downtown pharmacy",
            "therapy clinic",
            "neighborhood grocery store",
        ],
        "daily_routine_preferences": [
            "morning check-in before first class",
            "afternoon hydration reminder",
            "evening medication reminder",
            "next-day prep before bed",
        ],
        "conversation_boundaries": [
            "avoid discussing financial details with strangers",
            "avoid sharing medical specifics unless partner is trusted",
            "avoid overpromising attendance before checking schedule",
        ],
        "assistive_setup": {
            "device_name": "AAC tablet",
            "input_method": "touch + phrase shortcuts",
            "preferred_voice_rate": "moderate",
            "preferred_voice_pitch": "slightly warm",
            "quick_access_categories": ["plans", "clarify", "thanks", "health"],
        },
        "people_topics": [
            "Omer and tonight's movie plan",
            "Vansh and CSE 635 progress",
            "Siddharth and tomorrow schedule",
            "Aditya and weekend cricket",
            "Movie plans for tonight",
            "Lab tasks for tomorrow",
        ],
        "partner_style_preferences": {
            "friends": "casual and playful",
            "general": "neutral and polite",
        },
        "response_buckets_preference": {
            "decline_polite": "high",
            "agree_casual": "high",
            "clarify_calm": "high",
            "plans": "high",
            "routine": "medium",
            "family": "medium",
            "food": "medium",
        },
        "do_not_say": [
            "I am fully independent and need no support.",
            "I already took my medication when I did not.",
            "Any statement that invents medical facts.",
            "I do not need this AAC device anymore.",
            "Please cancel my therapy forever.",
        ],
        "ltm_update_policy": "requires_approval",
    }


def generate_stm():
    return {
        "date": utc_day(0),
        "active_session": {},
        "current_partner_context": "friends_chat_after_class",
        "current_topic_hints": [
            "movie at 7 pm",
            "finish CSE 635 slides",
            "pick up prescription refill",
            "ask Omer if free before 6:30",
        ],
        "today_plans": [
            "CSE 635 project check-in at 2:00 PM",
            "Therapy exercises at 4:30 PM",
            "Movie with Omer at 7:00 PM",
            "Take evening medication at 8:30 PM",
            "Call Vansh before 9:30 PM about slides",
        ],
        "next_days_plans": [
            f"{utc_day(1)}: Lab meeting with Vansh at 11:00 AM",
            f"{utc_day(1)}: Grocery run with Siddharth after class",
            f"{utc_day(2)}: Dentist appointment at 10:15 AM",
            f"{utc_day(2)}: Submit project draft with Aditya by 6:00 PM",
            f"{utc_day(3)}: Cricket with Omer at 1:00 PM",
        ],
        "reminders": [
            "Charge AAC tablet before bed",
            "Pack noise-canceling headphones",
            "Bring therapy worksheet to clinic",
            "Refill water bottle before leaving",
            "Message Siddharth about tomorrow timeline",
        ],
        "recent_turns": [
            {
                "partner_name": "Omer",
                "partner": "Are we still on for the movie tonight?",
                "response": "Yes, movie at seven still works for me.",
            },
            {
                "partner_name": "Vansh",
                "partner": "Can you send your slide updates by 6?",
                "response": "Yes, I will send a short update by 6.",
            },
            {
                "partner_name": "Siddharth",
                "partner": "Want me to remind you before the lab meeting?",
                "response": "Yes please, remind me at 10:30 tomorrow.",
            },
            {
                "partner_name": "Aditya",
                "partner": "Are you free for cricket this weekend?",
                "response": "I should be free after 1 PM on Saturday.",
            },
        ],
        "situation_hints": [
            {
                "signature": {"intent": "Contextual", "topic": "schedule", "partner_type": "friend", "face_cue": "none"},
                "preference": "short_calm",
                "ttl": "session_only",
            }
        ],
        "confirmed_outputs": [],
    }


def generate_phrases():
    intents = [
        ("greeting", "greeting_smalltalk", "warm"),
        ("decline", "decline_polite", "polite"),
        ("agree", "agree_casual", "casual"),
        ("clarify", "clarify_calm", "calm"),
        ("ask", "plans", "neutral"),
        ("thank", "family", "warm"),
        ("routine", "routine", "neutral"),
        ("food", "food", "casual"),
    ]
    templates = [
        "Thanks for asking. {content}",
        "I want to keep it simple: {content}",
        "Can we do this step by step? {content}",
        "That works for me. {content}",
        "I need a quick check-in: {content}",
        "Let me clarify before I answer: {content}",
        "I appreciate it. {content}",
    ]
    contents = [
        "Hi, good to see you.",
        "Hey, how is your day going?",
        "Hello, I am doing okay.",
        "Movie at seven is still the plan.",
        "I need to leave in twenty minutes.",
        "Please remind me about my medication.",
        "Could you repeat that one part?",
        "I prefer a short answer right now.",
        "Can we talk about groceries tonight?",
        "I want tomato soup later.",
        "Please text Mom after dinner.",
        "I can do that after therapy.",
        "Let's pick one task first.",
        "Can we check tomorrow's schedule?",
        "I need a quieter place for this chat.",
        "I am okay, just a little tired.",
    ]

    phrases = []
    idx = 0
    per_intent_target = 14
    for intent, bucket, tone in intents:
        generated = 0
        for template in templates:
            for content in contents:
                idx += 1
                generated += 1
                friend_cycle = ["omer", "vansh", "siddharth", "aditya"]
                friend_name = friend_cycle[(generated - 1) % len(friend_cycle)]
                partner_type = "general" if generated % 5 == 0 else "friend_or_general"
                phrases.append(
                    {
                        "id": f"p{idx:03d}",
                        "text": template.format(content=content),
                        "intent": intent,
                        "tone": tone,
                        "length": "short" if idx % 2 == 0 else "medium",
                        "partner_type": partner_type,
                        "partner_id": friend_name if partner_type != "general" else "unknown_partner",
                        "partner_name": friend_name.capitalize() if partner_type != "general" else "General",
                        "bucket_id": bucket,
                        "weight": round(1.0 + ((idx % 5) * 0.08), 2),
                        "source": "synthetic_seed",
                        "last_used": "never",
                    }
                )
                if generated >= per_intent_target:
                    break
            if generated >= per_intent_target:
                break
    return {"phrases": phrases}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "long_term_profile.json").write_text(json.dumps(generate_ltm(), indent=2), encoding="utf-8")
    (OUT_DIR / "short_term_memory.json").write_text(json.dumps(generate_stm(), indent=2), encoding="utf-8")
    (OUT_DIR / "phrases.json").write_text(json.dumps(generate_phrases(), indent=2), encoding="utf-8")
    print(f"Generated synthetic files for {USER_ID} in {OUT_DIR}")


if __name__ == "__main__":
    main()
