import re


DE = '[a-zäöüß]+'

LINK_TYPES = {
    # more complex types come first so that types that use subpatterns
    # are not matched prematurely;
    # e.g., _(x)_+xx_ should come before _+xx_ because otherwise
    # _(x)_+xx_ type links will be DEfined as _+xx_ ones
    "addition_with_expansion": f'_\({DE}\)_\+{DE}_',
    "addition_with_umlaut": f'_\+={DE}_',
    "replacement": f'_\-{DE}_\+{DE}_',
    "addition": f'_\-{DE}_\+{DE}_',
    "deletion_nom": f'_\-{DE}_',
    "deletion_non_nom": f'_#{DE}_',
    "umlaut": '_\+=_',
    "hyphen": '_--_',
    "concatenation": '_'
}


# def _addition_with_expansion(component, link):
#     link_clean = re.sub('[()+]', '', link)
#     return component + link_clean


# def _addition_with_umlaut(component, link):
#     pass


# def _replacement(component, link):
#     to_remove = re.search(LINK_TYPES["deletion_nom"], link, flags=re.I)
#     to_remove = re.search(LINK_TYPES["deletion_nom"], link, flags=re.I)