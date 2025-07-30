import params

def lean_str(lean):
    """
    Convert a state lean value to a string representation (ex. D+1.2, R+11.2)
    """
    if lean > 0:
        return f"D+{abs(lean * 100):.1f}"
    elif lean < 0:
        return f"R+{abs(lean * 100):.1f}"
    else:
        return "T+0.0"

def emoji_from_lean(
    lean,
    use_swing=False,
    SWING_LEAN=params.SWING_MARGIN,
    use_super_swing=False,
    SUPER_SWING_LEAN=params.SUPER_SWING_MARGIN
):
    """
    Get an emoji representation of the state lean.
    If use_super_swing is True, use a smaller threshold and a different emoji for super swing states.
    """
    # Convert to float if it's a string
    if isinstance(lean, str):
        try:
            lean = float(lean)
        except (ValueError, TypeError):
            return "❓"  # Unknown for invalid values
    
    if use_super_swing and abs(lean) <= SUPER_SWING_LEAN:
        return "⚪"  # White for super swing
    threshold = SWING_LEAN if use_swing else 0.0
    if lean > threshold:
        return "🔵"  # Blue for Democratic lean
    elif lean < -threshold:
        return "🔴"  # Red for Republican lean
    else:
        return "🟣"  # Purple for swing