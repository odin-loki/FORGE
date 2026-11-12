#!/usr/bin/env python3
"""Regenerates goldens.json: chat templates rendered by the real jinja2, set up the way transformers'
apply_chat_template sets it up (ImmutableSandboxedEnvironment, trim_blocks, lstrip_blocks, loop controls, its own
tojson, raise_exception, strftime_now).

    pip install jinja2
    python3 tests/data/chat_template/make_goldens.py [--check path/to/jdump]

The model templates below were written out from memory of the published tokenizer_config.json files; they are meant
to be faithful but have not been diffed against the originals. What the goldens pin down is that FORGE renders these
texts exactly as jinja2 does. The "feat_*" templates are synthetic and exercise one language feature each.
"""
import json, os, subprocess, sys
from datetime import datetime

from jinja2.exceptions import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))

T = {}
T["chatml"] = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"

T["qwen2_5"] = r"""{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""

T["qwen3"] = r"""{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "system" %}
        {{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
"""

T["llama2"] = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

T["llama3"] = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"

T["llama3_1"] = r"""{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message + builtin tools #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- "<|python_tag|>" + tool_call.name + ".call(" }}
            {%- for arg_name, arg_val in tool_call.arguments | items %}
                {{- arg_name + '="' + arg_val + '"' }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
                {%- endfor %}
            {{- ")" }}
        {%- else  %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {#- This means we're in ipython mode #}
            {{- "<|eom_id|>" }}
        {%- else %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""

T["mistral_v1"] = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

T["mistral_v3"] = r"""{%- if messages[0]["role"] == "system" %}
    {%- set system_message = messages[0]["content"] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}
{%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}

{#- This block checks for alternating user/assistant messages, skipping tool calling messages #}
{%- set ns = namespace() %}
{%- set ns.index = 0 %}
{%- for message in loop_messages %}
    {%- if not (message.role == "tool" or message.role == "tool_results" or (message.tool_calls is defined and message.tool_calls is not none)) %}
        {%- if (message["role"] == "user") != (ns.index % 2 == 0) %}
            {{- raise_exception("After the optional system message, conversation roles must alternate user/assistant/user/assistant/...") }}
        {%- endif %}
        {%- set ns.index = ns.index + 1 %}
    {%- endif %}
{%- endfor %}

{{- bos_token }}
{%- for message in loop_messages %}
    {%- if message["role"] == "user" %}
        {%- if tools is not none and (message == user_messages[-1]) %}
            {{- "[AVAILABLE_TOOLS] [" }}
            {%- for tool in tools %}
                {%- set tool = tool.function %}
                {{- '{"type": "function", "function": {' }}
                {%- for key, val in tool.items() if key != "return" %}
                    {%- if val is string %}
                        {{- '"' + key + '": "' + val + '"' }}
                    {%- else %}
                        {{- '"' + key + '": ' + val|tojson }}
                    {%- endif %}
                    {%- if not loop.last %}
                        {{- ", " }}
                    {%- endif %}
                {%- endfor %}
                {{- "}}" }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- else %}
                    {{- "]" }}
                {%- endif %}
            {%- endfor %}
            {{- "[/AVAILABLE_TOOLS]" }}
        {%- endif %}
        {%- if loop.last and system_message is defined %}
            {{- "[INST] " + system_message + "\n\n" + message["content"] + "[/INST]" }}
        {%- else %}
            {{- "[INST] " + message["content"] + "[/INST]" }}
        {%- endif %}
    {%- elif message.tool_calls is defined and message.tool_calls is not none %}
        {{- "[TOOL_CALLS] [" }}
        {%- for tool_call in message.tool_calls %}
            {%- set out = tool_call.function|tojson %}
            {{- out[:-1] }}
            {%- if not tool_call.id is defined or tool_call.id|length != 9 %}
                {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
            {%- endif %}
            {{- ', "id": "' + tool_call.id + '"}' }}
            {%- if not loop.last %}
                {{- ", " }}
            {%- else %}
                {{- "]" + eos_token }}
            {%- endif %}
        {%- endfor %}
    {%- elif message["role"] == "assistant" %}
        {{- " " + message["content"]|trim + eos_token}}
    {%- elif message["role"] == "tool_results" or message["role"] == "tool" %}
        {%- if message.content is defined and message.content.content is defined %}
            {%- set content = message.content.content %}
        {%- else %}
            {%- set content = message.content %}
        {%- endif %}
        {{- '[TOOL_RESULTS] {"content": ' + content|string + ", " }}
        {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}
            {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
        {%- endif %}
        {{- '"call_id": "' + message.tool_call_id + '"}[/TOOL_RESULTS]' }}
    {%- else %}
        {{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}
    {%- endif %}
{%- endfor %}
"""

T["gemma"] = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}"

T["phi3"] = "{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>\\n' + message['content'] + '<|end|>\\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\\n' + message['content'] + '<|end|>\\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\\n' + message['content'] + '<|end|>\\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% else %}{{ eos_token }}{% endif %}"

T["deepseek"] = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\\n\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"

T["deepseek_r1"] = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='', is_first_sp=true) %}{%- for message in messages %}{%- if message['role'] == 'system' %}{%- if ns.is_first_sp %}{% set ns.system_prompt = ns.system_prompt + message['content'] %}{% set ns.is_first_sp = false %}{%- else %}{% set ns.system_prompt = ns.system_prompt + '\\n\\n' + message['content'] %}{%- endif %}{%- endif %}{%- endfor %}{{ bos_token }}{{ ns.system_prompt }}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<\uff5cUser\uff5c>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<\uff5cAssistant\uff5c>' + content + '<\uff5cend\u2581of\u2581sentence\uff5c>'}}{%- endif %}{%- endfor -%}{% if add_generation_prompt and not ns.is_tool %}{{'<\uff5cAssistant\uff5c><think>\\n'}}{% endif %}"

T["zephyr"] = """{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<|user|>\\n' + message['content'] + eos_token }}
{% elif message['role'] == 'system' %}
{{ '<|system|>\\n' + message['content'] + eos_token }}
{% elif message['role'] == 'assistant' %}
{{ '<|assistant|>\\n'  + message['content'] + eos_token }}
{% endif %}
{% if loop.last and add_generation_prompt %}
{{ '<|assistant|>' }}
{% endif %}
{% endfor %}"""

# ---- synthetic feature templates
T["feat_loop"] = """{% for m in messages %}{{ loop.index }}/{{ loop.length }} {{ loop.index0 }} {{ loop.revindex }} {{ loop.revindex0 }} {{ loop.first }} {{ loop.last }} {{ loop.cycle('a', 'b', 'c') }} prev={{ loop.previtem.role if loop.previtem is defined else '-' }} next={{ loop.nextitem.role if loop.nextitem is defined else '-' }}
{% else %}none
{% endfor %}{% for m in messages if m.role == 'user' %}[{{ loop.index }}:{{ m.content|length }}]{% endfor %}
{% for m in [] %}x{% else %}empty{% endfor %}
{% for i in range(10) %}{% if i == 2 %}{% continue %}{% endif %}{% if i > 5 %}{% break %}{% endif %}{{ i }}{% endfor %}
{% for a, b in [[1, 2], [3, 4]] %}{{ a + b }};{% endfor %}{% for k, v in {'x': 1, 'y': [2]}.items() %}{{ k }}={{ v }},{% endfor %}{% for k in {'p': 1, 'q': 2} %}{{ k }}{% endfor %}{% for c in 'héy' %}<{{ c }}>{% endfor %}
{% for i in range(2) %}{% for j in range(3, 0, -1) %}{{ i }}{{ j }}{{ loop.index }} {% endfor %}{% endfor %}"""

T["feat_expr"] = """{{ 1 + 2 * 3 }} {{ (1 + 2) * 3 }} {{ 7 // 2 }} {{ -7 // 2 }} {{ 7 % 3 }} {{ -7 % 3 }} {{ 7 / 2 }} {{ 2 ** 10 }} {{ 1.5 + 1 }} {{ 3 * 1.0 }} {{ 10 / 5 }} {{ -3 }} {{ 1e3 }} {{ 0.1 + 0.2 }}
{{ 'a' ~ 1 ~ none ~ true ~ 2.5 }} {{ 'ab' * 3 }} {{ 'x' + 'y' }} {{ [1, 2] + [3] }} {{ 1 < 2 }} {{ 'a' < 'b' }} {{ 2 >= 2.0 }} {{ 1 == 1.0 }} {{ 'a' == 1 }} {{ [1, 'a'] == [1, 'a'] }} {{ {'a': 1} == {'a': 1} }}
{{ 'ell' in 'hello' }} {{ 3 in [1, 2, 3] }} {{ 'k' in {'k': 1} }} {{ 4 not in [1] }} {{ not true }} {{ true and 'yes' }} {{ false or 'no' }} {{ none or 0 or '' or 'last' }} {{ 1 and 2 }} {{ '' and 2 }}|
{{ 'a' if 1 > 2 else 'b' }} {{ 'only' if false }}| {{ ('x' if messages else 'y')|upper }} {{ 1 if messages|length > 1 else 2 if messages|length == 1 else 3 }}
{{ [1, 2, 3][0] }} {{ [1, 2, 3][-1] }} {{ [1, 2, 3, 4, 5][1:3] }} {{ [1, 2, 3, 4, 5][::-1] }} {{ [1, 2, 3, 4, 5][::2] }} {{ [1, 2, 3, 4, 5][-2:] }} {{ [1, 2, 3, 4, 5][:-2] }} {{ [1, 2, 3, 4, 5][3:1:-1] }} {{ 'héllo'[1:3] }} {{ 'héllo'[::-1] }} {{ 'abc'[1] }} {{ 'abc'[-1] }}
{{ {'a': {'b': [10, 20]}}.a.b[1] }} {{ {'a': 1}['a'] }} {{ {'a': 1}.zzz is defined }} {{ messages[0].nope is defined }} {{ nothing is defined }} {{ nothing is undefined }}{{ nothing }}|{{ messages[99] is defined }}
{{ none }} {{ true }} {{ false }} {{ [none, true, 1.0, 'q', "it's", 'say "hi"'] }} {{ {'k': 'v', 'n': [1, {'z': none}]} }} {{ () }} {{ (1, 2) }} {{ 'a' 'b' }} {{ "tab\\tnew\\\\n" }}
{{ 5 is odd }} {{ 4 is even }} {{ 9 is divisibleby 3 }} {{ 'a' is string }} {{ 1 is number }} {{ 1 is integer }} {{ 1.0 is float }} {{ {} is mapping }} {{ [] is iterable }} {{ 'a' is iterable }} {{ none is none }} {{ 1 is not none }} {{ true is boolean }} {{ true is true }} {{ 0 is false }} {{ 3 is eq 3 }} {{ 3 is gt 2 }} {{ 'a' is in ['a'] }} {{ [] is sequence }} {{ 1 is sameas 1 }} {{ 2 is lt(3) }} {{ 2 is ne 2 }}"""

T["feat_filters"] = """{{ '  pad  '|trim }}|{{ 'xxhixx'|trim('x') }}|{{ 'Hello World'|upper }}|{{ 'Hello World'|lower }}|{{ 'hello wORLD foo-bar'|title }}|{{ 'hELLO wORLD'|capitalize }}|{{ 'héllo'|length }}|{{ [1, 2, 3]|length }}|{{ {'a': 1}|length }}|{{ [1, 2]|count }}
{{ [1, 2, 3]|join(', ') }}|{{ ['a', 'b']|join }}|{{ messages|join('/', attribute='role') }}|{{ [1, 2, 3]|first }}|{{ [1, 2, 3]|last }}|{{ 'abc'|first }}|{{ 'abc'|list }}|{{ 'abc'|reverse }}|{{ [1, 2, 3]|reverse|list }}
{{ nothing|default('dflt') }}|{{ ''|default('dflt') }}|{{ ''|default('dflt', true) }}|{{ none|default('dflt') }}|{{ 0|d('z', true) }}|{{ 'v'|default('dflt') }}|{{ nothing|default }}|
{{ '42'|int + 1 }}|{{ '3.7'|int }}|{{ 'x'|int }}|{{ 'x'|int(9) }}|{{ 3.9|int }}|{{ '2.5'|float }}|{{ 3|float }}|{{ 'x'|float }}|{{ 5|string + '!' }}|{{ 2.5|round }}|{{ 3.5|round }}|{{ 3.14159|round(2) }}|{{ -3|abs }}|{{ -2.5|abs }}
{{ 'a-b-c'|replace('-', '+') }}|{{ 'aaa'|replace('a', 'b', 2) }}|{{ '<b> & "q"'|e }}|{{ '<b>'|safe }}|{{ [3, 1, 2]|sort }}|{{ ['b', 'A', 'c']|sort }}|{{ [3, 1, 2]|sort(reverse=true) }}|{{ [1, 2, 1, 3, 2]|unique|list }}|{{ ['a', 'A', 'b']|unique|list }}|{{ [3, 1, 2]|min }}|{{ [3, 1, 2]|max }}|{{ [1, 2, 3]|sum }}
{{ {'b': 1, 'a': 2}|items|list }}|{{ {'b': 1, 'A': 2}|dictsort }}|{{ messages|map(attribute='role')|list }}|{{ messages|map(attribute='role')|join(',') }}|{{ ['a', 'b']|map('upper')|list }}|{{ messages|selectattr('role', 'equalto', 'user')|list|length }}|{{ messages|rejectattr('role', 'eq', 'user')|map(attribute='role')|list }}|{{ [0, 1, '', 'a']|select|list }}|{{ [1, 2, 3, 4]|reject('odd')|list }}|{{ [1, 2, 3, 4]|select('gt', 2)|list }}|{{ messages|selectattr('name')|list|length }}
{{ 'line1\\nline2\\n\\nline4'|indent }}|{{ 'a\\nb'|indent(2, true) }}|{{ 'a\\n\\nb'|indent(width=1, blank=true) }}|{{ 'a\\nb'|indent(first=true) }}
{{ {'s': 'a"b\\\\c\\n\\té', 'n': none, 't': true, 'f': 1.5, 'i': 3, 'l': [1, [2, {}], []], 'e': {}}|tojson }}
{{ {'a': [1, {'b': 'c'}], 'd': {}}|tojson(indent=2) }}
{{ messages|tojson }}
{{ 'x'|upper|lower|capitalize|length }} {{ ('a,b,c'.split(',')|last) ~ ('  x '|trim) }} {{ -5|abs + 1 }} {{ 'a b'.split()|length * 2 }}"""

T["feat_methods"] = """{{ '  x  '.strip() }}|{{ '  x  '.lstrip() }}|{{ '  x  '.rstrip() }}|{{ 'xxaxx'.strip('x') }}|{{ 'abc'.upper() }}|{{ 'ABC'.lower() }}|{{ "they're here".title() }}|{{ 'hELLO'.capitalize() }}
{{ 'abc'.startswith('ab') }}|{{ 'abc'.endswith('bc') }}|{{ 'abc'.startswith('x') }}|{{ 'a-b'.replace('-', '') }}|{{ 'aXbXc'.split('X') }}|{{ ' a  b '.split() }}|{{ 'a,b,c'.split(',', 1) }}|{{ 'héllo wörld'.find('w') }}|{{ 'abc'.find('z') }}|{{ 'banana'.count('an') }}|{{ ', '.join(['a', 'b']) }}|{{ 'a\\nb\\n'.splitlines() }}|{{ '12'.isdigit() }}
{{ {'a': 1}.get('a') }}|{{ {'a': 1}.get('b') }}|{{ {'a': 1}.get('b', 'dflt') }}|{{ {'a': 1, 'b': 2}.keys()|list }}|{{ {'a': 1, 'b': 2}.values()|list }}|{{ {'a': 1}.items()|list }}|{{ dict(a=1, b='x') }}
{{ messages[0]['content'].split(' ')[0] }}|{{ messages[-1].role.upper() }}|{{ (messages|first).content|length }}"""

T["feat_scope"] = """{% set x = 1 %}{% set ns = namespace(count=0, items='') %}{% for m in messages %}{% set x = x + 1 %}{% set ns.count = ns.count + 1 %}{% set ns.items = ns.items ~ m.role[0] %}{{ x }}{% endfor %} x={{ x }} count={{ ns.count }} items={{ ns.items }}
{% if true %}{% set y = 'set in if' %}{% endif %}{{ y }}
{% set block %}captured {{ x }} {% for i in range(3) %}{{ i }}{% endfor %}{% endset %}[{{ block }}] {{ block|length }}
{% macro greet(name, punct='!') %}Hello {{ name }}{{ punct }}{% endmacro %}{{ greet('A') }} {{ greet('B', '?') }} {{ greet(name='C', punct='.') }} {{ greet('D')|upper }}
{% macro wrap(m) -%}
    <{{ m.role }}>{{ m.content|trim }}</{{ m.role }}>
{%- endmacro %}{% for m in messages %}{{ wrap(m) }}{% endfor %}
{% set a, b = 1, 2 %}{{ a }}{{ b }}{% set t = (1, 'two') %}{{ t[1] }}"""

T["feat_whitespace"] = """line1
  {% if true %}
    indented body
  {% endif %}
  {{ 'expr keeps indentation' }}
{%- if true %}   lstrip-minus   {%- endif %}
A {{- ' glued ' -}} B
{% if true -%}
     trimmed after
{%- endif %}
  {%+ if true %}plus keeps indentation{% endif %}
{% if true +%}
plus keeps newline{% endif %}
{# comment #}
  {# indented comment #}
after comments {#- tight -#} end
{% raw %}{{ not rendered }} {% neither %}{% endraw %}
{{ '}}' }} {{ "%}" }} {{ {'a': {'b': 1}}['a'] }}{{ '{{' }}
tail  """

MSG_BASIC = [{"role": "user", "content": "Hello!"}]
MSG_SYS = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hi, what is 2+2?"},
           {"role": "assistant", "content": "  4.  "}, {"role": "user", "content": "And 3+3?\nAnswer briefly."}]
MSG_TURNS = [{"role": "user", "content": "Bonjour, ça va ? 你好"}, {"role": "assistant", "content": "Oui !"}, {"role": "user", "content": "Tr\u00e8s bien \U0001F600"}]
MSG_BAD_ORDER = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
MSG_THINK = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "<think>\nreasoning here\n</think>\n\nanswer one"}, {"role": "user", "content": "q2"},
             {"role": "assistant", "content": "plain answer", "reasoning_content": "kept apart"}]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city.",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. \"Paris\""},
                                                                                             "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                                                                                             "days": {"type": "integer", "minimum": 1}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "now", "description": "Current time", "parameters": {"type": "object", "properties": {}}}}]
MSG_TOOLCALL = [{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "abc123xyz", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}}]},
                {"role": "tool", "tool_call_id": "abc123xyz", "name": "get_weather", "content": "{\"temp\": 21.5, \"sky\": \"clear\"}"},
                {"role": "assistant", "content": "It is 21.5 degrees and clear."}, {"role": "user", "content": "Thanks"}]
MSG_TWO_TOOLS = [{"role": "user", "content": "Both"},
                 {"role": "assistant", "content": None, "tool_calls": [{"id": "call00001", "type": "function", "function": {"name": "now", "arguments": {}}},
                                                                         {"id": "call00002", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Oslo", "days": 2}}}]},
                 {"role": "tool", "tool_call_id": "call00001", "content": "12:00"}, {"role": "tool", "tool_call_id": "call00002", "content": "rain"}]

SPECIAL = {"chatml": ("", "<|im_end|>"), "qwen2_5": ("", "<|im_end|>"), "qwen3": ("", "<|im_end|>"), "llama2": ("<s>", "</s>"), "llama3": ("<|begin_of_text|>", "<|eot_id|>"),
           "llama3_1": ("<|begin_of_text|>", "<|eot_id|>"), "mistral_v1": ("<s>", "</s>"), "mistral_v3": ("<s>", "</s>"), "gemma": ("<bos>", "<eos>"), "phi3": ("<s>", "<|endoftext|>"),
           "deepseek": ("<\uff5cbegin\u2581of\u2581sentence\uff5c>", "<\uff5cend\u2581of\u2581sentence\uff5c>"), "deepseek_r1": ("<\uff5cbegin\u2581of\u2581sentence\uff5c>", "<\uff5cend\u2581of\u2581sentence\uff5c>"),
           "zephyr": ("<s>", "</s>")}


def environment():
    def raise_exception(message): raise TemplateError(message)
    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    def strftime_now(format): return datetime.now().strftime(format)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env


def main():
    env = environment()
    cases = []
    def add(template, messages, add_generation_prompt=True, tools=None, extra=None):
        bos, eos = SPECIAL.get(template, ("<s>", "</s>"))
        case = {"template": template, "messages": messages, "add_generation_prompt": add_generation_prompt, "bos_token": bos, "eos_token": eos}
        if tools is not None: case["tools"] = tools
        if extra: case["extra"] = extra
        kw = dict(messages=messages, add_generation_prompt=add_generation_prompt, bos_token=bos, eos_token=eos, **(extra or {}))
        if tools is not None: kw["tools"] = tools
        try: case["expected"] = env.from_string(T[template]).render(**kw)
        except TemplateError as e: case["error"] = str(e)
        except Exception as e: case["error"] = type(e).__name__
        cases.append(case)

    for name in SPECIAL:
        for msgs in (MSG_BASIC, MSG_SYS, MSG_TURNS, MSG_BAD_ORDER):
            for agp in (True, False): add(name, msgs, agp)
    for name in ("qwen2_5", "llama3_1", "mistral_v3"):
        for msgs in (MSG_BASIC, MSG_SYS, MSG_TOOLCALL, MSG_TWO_TOOLS):
            add(name, msgs, True, TOOLS)
        add(name, MSG_TOOLCALL, True)
        add(name, MSG_TWO_TOOLS, False)
    add("llama3_1", MSG_TOOLCALL, True, TOOLS, {"tools_in_user_message": False, "date_string": "12 Nov 2026"})
    add("llama3_1", MSG_SYS, True, None, {"builtin_tools": ["brave_search", "code_interpreter", "wolfram_alpha"]})
    add("llama3_1", [], True, TOOLS)
    add("mistral_v3", [{"role": "user", "content": "x"}, {"role": "assistant", "content": None, "tool_calls": [{"id": "short", "function": {"name": "now", "arguments": {}}}]}], True, TOOLS)
    for msgs in (MSG_THINK, MSG_SYS, MSG_BASIC):
        add("qwen3", msgs, True, None, {"enable_thinking": False}); add("qwen3", msgs, True, None, {"enable_thinking": True})
    add("deepseek_r1", MSG_THINK); add("chatml", []); add("llama3", []); add("gemma", MSG_SYS)
    for name in (n for n in T if n.startswith("feat_")):
        for msgs in (MSG_SYS, MSG_BASIC, []): add(name, msgs)

    with open(os.path.join(HERE, "goldens.json"), "w", encoding="utf-8") as f:
        json.dump({"templates": T, "cases": cases}, f, ensure_ascii=False, indent=0, separators=(",", ":"))
    print(len(T), "templates,", len(cases), "cases,", sum("error" in c for c in cases), "expected errors")
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    if "--check" in args:   # jdump goldens.json prints one JSON string (or {"error": ...}) per case
        r = subprocess.run([args["--check"], os.path.join(HERE, "goldens.json")], capture_output=True, text=True)
        if r.returncode: print("jdump failed:", r.stderr[-2000:]); return
        bad = 0
        for c, line in zip(cases, r.stdout.splitlines()):
            got = json.loads(line)
            if "error" in c:
                if not (isinstance(got, dict) and (c["error"] in got["error"] or not c["error"][:1].isupper() or "Error" in c["error"])): bad += 1; print("ERR", c["template"], "want error", c["error"], "got", got)
            elif got != c["expected"]:
                bad += 1
                print("DIFF", c["template"], len(c["messages"]), c["add_generation_prompt"], c.get("extra"))
                exp = c["expected"]; g = got if isinstance(got, str) else json.dumps(got)
                i = next((k for k in range(min(len(exp), len(g))) if exp[k] != g[k]), min(len(exp), len(g)))
                print("   want", repr(exp[max(0, i - 60):i + 80])); print("   got ", repr(g[max(0, i - 60):i + 80]))
        print("mismatches:", bad)


if __name__ == "__main__":
    main()
