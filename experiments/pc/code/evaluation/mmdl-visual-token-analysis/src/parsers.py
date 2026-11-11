"""Frozen scoring: official rule excerpts + target final-answer parser + judge-output parser.

Official functions copied verbatim from QwenLM/Qwen3-VL, commit
96588727e44c78b25ba03ea03b8e12f7e64fd0da, evaluation/mmmu/eval_utils.py.
See THIRD_PARTY_LICENSES.txt. Other functions are project-specific.
"""
import ast
import copy
import re
import string

TARGET_VERSION = 'terminal-final-field-first'
JUDGE_VERSION = VERSION = 'judge-terminal-decision'

def can_infer_option(answer, choices):
    """Rule-based extraction of answer option."""
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = copy.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3:
                # print(f'A might be a quantifier in the string: {answer}.')
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False

def can_infer_text(answer, choices):
    """Extract answer by matching text content."""
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False

def can_infer(answer, choices):
    """Combined approach to infer answer choice."""
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)

def normalize_line(line):
    # Markdown-only normalization, no semantic/numeric rewriting.
    text = line.strip()
    text = re.sub(r'^#{1,6}\s+', '', text)
    text = re.sub(r'^[✅✔☑]\ufe0f?\s*', '', text)
    for _ in range(3):
        old = text
        for marker in ('**','__','`','*','_'):
            if text.startswith(marker) and text.endswith(marker) and len(text)>2*len(marker):
                text = text[len(marker):-len(marker)].strip()
                break
        if text == old:
            break
    # Bold label only: **Final answer:** B.
    text = re.sub(r'^(?:\*\*|__)(Final\s+answer\s*:)(?:\*\*|__)\s*', r'\1 ', text, flags=re.I)
    return text

def mc_letter(payload, choices):
    text = normalize_line(payload)
    match = re.match(r'^(?:\(\s*([a-z])\s*\)|\[\s*([a-z])\s*\]|([a-z])(?=$|[\s.:,;)\-]))', text, re.I)
    if not match:
        return None, 'not_a_leading_option_letter'
    answer = next(x for x in match.groups() if x).upper()
    if answer not in choices:
        return None, 'invalid_option_letter'
    rest = text[match.end():].strip()
    # Do not turn "B or C", lists, or multiple alternatives into one answer.
    letters = {x.upper() for x in re.findall(r'(?<!\w)([A-Za-z])(?!\w)', rest)} & set(choices)
    if letters - {answer}:
        return None, 'multiple_option_letters_in_final_field'
    if re.match(r'^(?:or|and|/|\\|\|)', rest, re.I):
        return None, 'ambiguous_option_expression'
    return answer, None

def extract_final(raw, kind, choices):
    text = str(raw).split('</think>')[-1].strip()
    # Ignore only standalone Markdown fences; substantive trailing text remains.
    lines = [normalize_line(x) for x in text.splitlines() if x.strip()]
    lines = [x for x in lines if not re.fullmatch(r'(?:`{3,}|~{3,})(?:[A-Za-z0-9_-]+)?', x)]
    matches = []
    for i,line in enumerate(lines):
        m = re.fullmatch(r'Final\s+answer\s*:\s*(.*)', line, re.I)
        if m:
            payload = normalize_line(m.group(1))
            end = i
            if not payload and i+1 < len(lines):
                # A section heading immediately followed by an explicit field
                # is one field, not an answer literally equal to "Final answer: B".
                if re.fullmatch(r'Final\s+answer\s*:\s*.*',lines[i+1],re.I):
                    continue
                payload = normalize_line(lines[i+1])
                end = i+1
            matches.append({'line':i, 'end_line':end, 'payload':payload})
    if not matches:
        return {'status':'absent','reason':'no_explicit_final_field','fields':[]}
    if matches[-1]['end_line'] != len(lines)-1:
        return {'status':'ambiguous','reason':'substantive_text_after_final_field','fields':matches}
    answers = []
    for field in matches:
        if not field['payload']:
            return {'status':'ambiguous','reason':'empty_final_answer','fields':matches}
        if kind == 'multiple-choice':
            answer, failure = mc_letter(field['payload'], choices)
            if failure:
                return {'status':'ambiguous','reason':failure,'fields':matches}
        else:
            answer = re.sub(r'\s+', ' ', field['payload']).strip()
        answers.append(answer)
    if len(set(answers)) != 1:
        return {'status':'ambiguous','reason':'conflicting_final_answers','fields':matches}
    return {'status':'extracted','reason':'terminal_explicit_final_field',
            'answer':answers[-1], 'payload':matches[-1]['payload'], 'fields':matches,
            'repeated_same_answer':len(matches)>1}

def clean(line):
    text=line.strip()
    text=re.sub(r'^#{1,6}\s+','',text)
    text=re.sub(r'^[✅✔☑]\ufe0f?\s*','',text)
    for _ in range(3):
        prior=text
        for marker in ('**','__','`','*','_'):
            if text.startswith(marker) and text.endswith(marker) and len(text)>2*len(marker):
                text=text[len(marker):-len(marker)].strip()
                break
        if prior==text:
            break
    return re.sub(r'^(?:\*\*|__)((?:Your\s+output|Output|Final\s+answer)\s*:)(?:\*\*|__)\s*',r'\1 ',text,flags=re.I)

def parse_judge_output(raw,choices,official_rule):
    text=str(raw).split('</think>')[-1].strip()
    lines=[clean(x) for x in text.splitlines() if x.strip()]
    lines=[x for x in lines if not re.fullmatch(r'(?:`{3,}|~{3,})(?:[A-Za-z0-9_-]+)?',x)]
    last=lines[-1] if lines else ''
    match=re.fullmatch(r'(?:(?:Your\s+output|Output|Final\s+answer)\s*:\s*)?([A-Za-z])\s*\.?',last,re.I)
    if match:
        letter=match.group(1).upper()
        route='terminal_judge_decision'
    else:
        # Keep successfully supported official prose formats; terminal decisions
        # override whole-response substring/letter matches.
        letter=official_rule(text,copy.deepcopy(choices))
        route='official_judge_output_fallback'
    if letter=='Z':
        status='no_match'
    elif letter in choices:
        status='valid_option'
    else:
        status='invalid_option' if match else 'unparsed'
    return {'parser_version':VERSION,'status':status,
            'decision_parsed':status in ('valid_option','no_match'),
            'valid_option_extracted':status=='valid_option',
            'answer':letter if status in ('valid_option','no_match') else None,
            'candidate':letter or None,'route':route,'last_line':last}


def choice_info(row):
    values = row.get('options', [])
    values = ast.literal_eval(values) if isinstance(values, str) else values
    if row['question_type'] == 'open':
        choices = {'A': row['reference_answer'], 'B': 'Other Answers'}
    elif isinstance(values, dict):
        if list(values) != list(string.ascii_uppercase[:len(values)]):
            raise ValueError('Invalid option letters/order.')
        choices = copy.deepcopy(values)
    else:
        if not isinstance(values, list) or not 1 <= len(values) <= 26:
            raise ValueError('Invalid multiple-choice options.')
        choices = dict(zip(string.ascii_uppercase, values))
    # Deliberately preserve official normalization, including single-letter open GT.
    gold = row['reference_answer'] if row['reference_answer'] in list(string.ascii_uppercase) else 'A'
    return choices, gold


def grade_response(row):
    choices, gold = choice_info(row)
    response = row['model_response']
    result = {'status':'graded', 'grading_gold':gold, 'extracted_answer':None,
              'hit':0, 'correct':False, 'token_limit_forced_incorrect':False,
              'random_fallback':False}
    if response['finish_reason'] == 'length':
        result.update(method='output_limit_incorrect', reason='finish_reason_length_policy',
                      token_limit_forced_incorrect=True)
        return result
    field = extract_final(response['raw'], row['question_type'], choices)
    result['final_field'] = field
    if field['status'] == 'extracted':
        answer = field['answer'] if row['question_type']=='multiple-choice' else can_infer(field['payload'], copy.deepcopy(choices))
        method = 'explicit_final_parser'
        reason = 'explicit_final_answer_priority' if answer else 'open_final_answer_official_match_failed'
    elif field['status'] == 'absent':
        processed = str(response['raw']).split('</think>')[-1].strip()
        answer = can_infer(processed, copy.deepcopy(choices))
        method, reason = 'official_parser_fallback', field['reason']
    else:
        answer, method, reason = None, 'pending_judge', field['reason']
    if answer:
        hit = int(answer == gold)
        result.update(extracted_answer=answer, hit=hit, correct=bool(hit), method=method, reason=reason)
    else:
        result.update(status='pending_judge', hit=None, correct=None, method='pending_judge', reason=reason)
    return result


def grade_judge(row, raw):
    choices, gold = choice_info(row)
    decision = parse_judge_output(raw, choices, can_infer)
    hit = int(decision['valid_option_extracted'] and decision['answer'] == gold)
    return {'status':'graded', 'correct':bool(hit), 'hit':hit,
        'grading_gold':gold, 'extracted_answer':decision['answer'],
        'method':'local_qwen3_8b_judge_parser', 'reason':decision['status'],
        'judge_parser_decision':decision, 'random_fallback':False,
        'token_limit_forced_incorrect':False,
        'extraction_failure_forced_incorrect':not decision['decision_parsed']}


def selfcheck():
    choices = {'A':'red','B':'green','C':'blue'}
    def field(text):
        return extract_final(text, 'multiple-choice', choices)
    assert field('Discuss A and C.\nFinal answer: B')['answer'] == 'B'
    assert field('### Final Answer:\nB')['answer'] == 'B'
    assert field('✅ **Final answer:** (b). green')['answer'] == 'B'
    assert field('Final answer: B. green\nFinal answer: B')['answer'] == 'B'
    for text in ('Final answer: A\nFinal answer: B','Final answer: B or C','Final answer: B\nMore text','Final answer: D'):
        assert field(text)['status'] == 'ambiguous'
    assert field('B. green')['status'] == 'absent'
    row = {'question_type':'multiple-choice','options':list(choices.values()),'reference_answer':'B',
        'model_response':{'raw':'Final answer: B','finish_reason':'length'}}
    assert grade_response(row)['hit'] == 0
    row['model_response']['finish_reason'] = 'stop'
    assert grade_response(row)['hit'] == 1
    row['model_response']['raw'] = 'B. green'
    assert grade_response(row)['method'] == 'official_parser_fallback'
    row['model_response']['raw'] = 'Final answer: A\nFinal answer: B'
    assert grade_response(row)['status'] == 'pending_judge'
    for text in ('Your output: B','Output: B','Final answer: B','**B**'):
        assert grade_judge(row,text)['hit'] == 1
    assert grade_judge(row,'Z')['judge_parser_decision']['status'] == 'no_match'
    assert grade_judge(row,'D')['extraction_failure_forced_incorrect']
    assert grade_judge(row,'A or B')['extraction_failure_forced_incorrect']
    opened = {'question_type':'open','options':[],'reference_answer':'42',
        'model_response':{'raw':'Final answer: 42','finish_reason':'stop'}}
    assert grade_response(opened)['hit'] == 1
    opened['reference_answer'] = 'C'; opened['model_response']['raw'] = 'Final answer: C'
    assert grade_response(opened)['hit'] == 0  # Preserve the audited official quirk.
    print('PASS: final-field ambiguity, terminal priority, official fallback, length policy, judge Z/invalid, official open GT.')
