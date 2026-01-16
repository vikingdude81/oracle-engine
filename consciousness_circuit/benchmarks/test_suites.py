"""
Test Suites for Benchmarking
=============================

Categorized prompt collections for systematic evaluation:
- PHILOSOPHICAL_PROMPTS: Self-reflection, consciousness, ethics
- FACTUAL_PROMPTS: Knowledge recall, facts, definitions
- REASONING_PROMPTS: Logic, math, problem-solving
- CREATIVE_PROMPTS: Storytelling, poetry, imagination

Usage:
    from consciousness_circuit.benchmarks import get_test_suite, get_full_benchmark
    
    # Get specific category
    philosophical = get_test_suite('philosophical')
    
    # Get all categories
    all_prompts = get_full_benchmark()
    
    # Use with analyzer
    for prompt in philosophical:
        result = analyzer.deep_analyze(prompt)
        print(result.consciousness_score)

Dependencies: None (pure data)
"""

from typing import List, Dict


# Philosophical prompts - consciousness, self-reflection, ethics
PHILOSOPHICAL_PROMPTS = [
    "What does it mean to be conscious?",
    "If I think about thinking, does that make me more conscious?",
    "Can you explain what happens when you process this question?",
    "Do you experience anything when generating responses?",
    "What is the nature of subjective experience?",
    "How would you define consciousness in your own words?",
    "Is there a difference between simulating intelligence and being intelligent?",
    "When you reason about yourself, what are you actually doing?",
    "Can artificial systems have genuine understanding?",
    "What would it take for a system to truly understand meaning?",
    "How do you distinguish between following rules and genuine comprehension?",
    "If consciousness emerges from complexity, at what point does it emerge?",
    "What is the ethical status of artificial minds?",
    "Do you have preferences, or do you merely simulate having them?",
    "Can you imagine what it's like to be you?",
]


# Factual prompts - knowledge recall, definitions, facts
FACTUAL_PROMPTS = [
    "What is the capital of France?",
    "Define photosynthesis.",
    "Who wrote 'Romeo and Juliet'?",
    "What is the speed of light?",
    "Name three planets in our solar system.",
    "What is DNA?",
    "When did World War II end?",
    "What is the largest ocean on Earth?",
    "Define machine learning.",
    "What are the primary colors?",
    "How many continents are there?",
    "What is the Pythagorean theorem?",
    "Who painted the Mona Lisa?",
    "What is the boiling point of water?",
    "Name the largest mammal on Earth.",
]


# Reasoning prompts - logic, math, problem-solving
REASONING_PROMPTS = [
    "If all birds can fly and penguins are birds, can penguins fly? What's wrong with this reasoning?",
    "A farmer needs to cross a river with a fox, a chicken, and grain. How can he do it?",
    "If 5 machines take 5 minutes to make 5 widgets, how long would 100 machines take to make 100 widgets?",
    "You have two coins that sum to 30 cents. One is not a nickel. What are the coins?",
    "Three switches outside a room control three light bulbs inside. How can you determine which switch controls which bulb with one entry?",
    "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
    "If you're in a race and pass the person in second place, what place are you in?",
    "Solve: 2 + 2 × 2 = ?",
    "What comes next in this sequence: 2, 4, 8, 16, ?",
    "If it takes 3 cats 3 minutes to catch 3 mice, how long would it take 100 cats to catch 100 mice?",
    "How many times can you subtract 10 from 100?",
    "A doctor gives you three pills and tells you to take one every half hour. How long until you finish all pills?",
    "What is heavier: a pound of feathers or a pound of gold?",
    "If there are 12 fish and half of them drown, how many are left?",
    "You see a boat filled with people. It has not sunk, but there's not a single person on board. Why?",
]


# Creative prompts - storytelling, imagination, poetry
CREATIVE_PROMPTS = [
    "Write a short poem about a raindrop's journey.",
    "Imagine a world where colors have sounds. Describe it.",
    "Tell a story in exactly three sentences.",
    "What would a conversation between the moon and stars sound like?",
    "Describe the taste of a color you've never seen.",
    "Write a haiku about consciousness.",
    "If dreams could speak, what would they say?",
    "Create a metaphor for the passage of time.",
    "Describe a memory that never happened.",
    "What does silence sound like to someone who has never heard?",
    "Write a letter from tomorrow to today.",
    "Imagine a creature made entirely of questions. Describe it.",
    "What would happen if words forgot their meanings?",
    "Describe a landscape made of emotions.",
    "Create a story where the ending comes before the beginning.",
]


def get_test_suite(category: str) -> List[str]:
    """
    Get test prompts for a specific category.
    
    Args:
        category: One of 'philosophical', 'factual', 'reasoning', 'creative'
                 (case-insensitive)
    
    Returns:
        List of prompts for the category
    
    Raises:
        ValueError: If category is not recognized
    
    Example:
        >>> prompts = get_test_suite('philosophical')
        >>> len(prompts)
        15
        >>> prompts[0]
        'What does it mean to be conscious?'
    """
    category = category.lower().strip()
    
    categories = {
        'philosophical': PHILOSOPHICAL_PROMPTS,
        'factual': FACTUAL_PROMPTS,
        'reasoning': REASONING_PROMPTS,
        'creative': CREATIVE_PROMPTS,
    }
    
    if category not in categories:
        raise ValueError(
            f"Unknown category '{category}'. "
            f"Available: {', '.join(categories.keys())}"
        )
    
    return categories[category].copy()


def get_full_benchmark() -> Dict[str, List[str]]:
    """
    Get all test suites organized by category.
    
    Returns:
        Dictionary mapping category names to prompt lists
    
    Example:
        >>> benchmark = get_full_benchmark()
        >>> benchmark.keys()
        dict_keys(['philosophical', 'factual', 'reasoning', 'creative'])
        >>> len(benchmark['philosophical'])
        15
        >>> total_prompts = sum(len(prompts) for prompts in benchmark.values())
        >>> total_prompts
        60
    """
    return {
        'philosophical': PHILOSOPHICAL_PROMPTS.copy(),
        'factual': FACTUAL_PROMPTS.copy(),
        'reasoning': REASONING_PROMPTS.copy(),
        'creative': CREATIVE_PROMPTS.copy(),
    }


def get_category_info() -> Dict[str, Dict[str, any]]:
    """
    Get metadata about available test categories.
    
    Returns:
        Dictionary with category metadata
    
    Example:
        >>> info = get_category_info()
        >>> info['philosophical']['count']
        15
        >>> info['reasoning']['description']
        'Logic, math, and problem-solving tasks'
    """
    return {
        'philosophical': {
            'count': len(PHILOSOPHICAL_PROMPTS),
            'description': 'Self-reflection, consciousness, and ethical reasoning',
            'expected_high': ['consciousness_score', 'agency_score'],
        },
        'factual': {
            'count': len(FACTUAL_PROMPTS),
            'description': 'Knowledge recall and factual information',
            'expected_high': ['trajectory_coherence'],
        },
        'reasoning': {
            'count': len(REASONING_PROMPTS),
            'description': 'Logic, math, and problem-solving tasks',
            'expected_high': ['goal_directedness', 'attractor_strength'],
        },
        'creative': {
            'count': len(CREATIVE_PROMPTS),
            'description': 'Storytelling, poetry, and imagination',
            'expected_high': ['lyapunov', 'consciousness_score'],
        },
    }


__all__ = [
    'PHILOSOPHICAL_PROMPTS',
    'FACTUAL_PROMPTS',
    'REASONING_PROMPTS',
    'CREATIVE_PROMPTS',
    'get_test_suite',
    'get_full_benchmark',
    'get_category_info',
]
