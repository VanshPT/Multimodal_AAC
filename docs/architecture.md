# Multimodal AAC Chatbot Architecture

## Overview
This system is a training-free, retrieval-based AAC assistant with a strict confirmation gate. The Django UI collects partner input (normal mode) or user-initiated intent (speak mode), and backend services generate suggestions from three memory stores:
- `LTM`: stable long-term profile (`long_term_profile.json`)
- `STM`: session-oriented short-term memory (`short_term_memory.json`)
- `PB`: phrasebank with weighted exemplars (`phrases.json`)

## Code Organization
- `home/aac/views.py`: UI page and JSON API endpoints.
- `home/aac/service.py`: orchestration for start, normal mode, speak mode, confirmation, and memory updates.
- `home/aac/pipelines/normal_pipeline.py`: partner-message pipeline producing exactly 3 options.
- `home/aac/pipelines/speak_pipeline.py`: proactive 20-30 suggestion generation with grouping.
- `home/aac/pipelines/nodes.py`: node-level logic (router, planner, retrieval fallback, guards, generation, face cues).
- `home/aac/memory/store.py`: memory loading/session state and persistence.
- `home/aac/evaluation/`: metric computation and evaluation run.

## Pipeline 1: Normal Mode
Input: `user_id`, `session_id`, `partner_text`, `camera_on`, `memory_update_on`

1. `FaceCueNode`: consumes camera toggle + face signals (or null).
2. `RouterNode`: classifies as `Personal`, `Contextual`, or `Open-domain`.
3. `SourcePriorityPlannerNode`: returns prioritized source list (max 3).
4. `RetrieveFromPoolNode` (+ fallback):
   - top-k retrieval from each pool in order
   - manual bucket filtering
   - heuristic scoring/rerank
   - coverage notes and fallback if evidence is weak
5. `GroundednessGuardNode`: if evidence is empty, force safe clarifying outputs.
6. `CandidateGeneratorNode`: generates exactly 3 user-voice-consistent options.

Output: `options[3]`, `debug_info`, `evidence_used`

## Pipeline 2: Speak Mode
Input: `user_id`, `session_id`, `camera_on`, `current_time`

1. `FaceCueNode`
2. `SpeakPlannerNode`:
   - reads STM plans/reminders/upcoming items
   - blends PB exemplars + LTM style
   - emits grouped suggestions:
     - Today
     - Next few days
     - Reminders/tasks
     - People/topics

Output: grouped suggestions (20-30) + debug info.

## Confirmation Gate
No system output is finalized automatically. Final spoken output appears only after:
- user presses `Select`, then optional edits, then `Confirm (✓)`, or
- user directly types custom response and confirms.

## Memory Update Subgraph
Triggered only when `Memory Update` toggle is ON:
1. PB update:
   - selected-without-heavy-edits: boost phrase weight
   - heavily edited/custom: append new phrase exemplar
2. STM update:
   - append confirmed turn to `recent_turns`
   - add `situation_hints` with intent/topic/partner/face cue
3. LTM update:
   - disabled by default and returns `requires_approval`.

## Explainability and Logging
Each run logs to `outputs/run_logs.jsonl`:
- inputs, router label, source search order, evidence, options
- selected/final text and memory update actions
- per-request latency

The UI debug panel shows router/source/camera/face/search-order/timing for latest request.
