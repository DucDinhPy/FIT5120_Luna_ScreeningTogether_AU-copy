<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'
import { askSupport, type SupportSource, type SupportTurn } from '@/api/supportChat'
import { useTranslation } from '@/i18n/useTranslation'

const props = defineProps<{ open: boolean }>()
const { t, currentLanguage } = useTranslation()
const turns = ref<(SupportTurn & { sources: SupportSource[] })[]>([])
const transcript = ref<HTMLElement>()
const question = ref('')
const pending = ref('')
const error = ref('')
let request: AbortController | undefined
onBeforeUnmount(() => request?.abort())

watch(
  [() => props.open, () => turns.value.length, pending],
  () => {
    if (props.open && transcript.value) transcript.value.scrollTop = transcript.value.scrollHeight
  },
  { flush: 'post' },
)

function normalizeQuestion(value: string) {
  return value.normalize('NFKC').toLowerCase().trim().replace(/[?؟]+$/, '').trim()
}

async function send() {
  const message = question.value.trim()
  if (!message || pending.value) return

  error.value = ''
  // shortcut: only exact FAQ matches, let Gemini handle paraphrases.
  for (const lng of new Set([currentLanguage.value, 'en'])) {
    const options = { ns: 'faq', lng }
    if (normalizeQuestion(message) === normalizeQuestion(t('questions.what.question', options))) {
      turns.value.push({
        question: message,
        answer: t('questions.what.answer', options),
        sources: [{ title: t('title', options), url: '/faq', publisher: null, page: null }],
      })
      question.value = ''
      return
    }
  }

  pending.value = message
  request = new AbortController()
  try {
    const history = turns.value.slice(-4).map(({ question, answer }) => ({
      question,
      answer: answer.slice(0, 6000),
    }))
    const reply = await askSupport(message, history, request.signal)
    if (reply) {
      turns.value.push({ question: message, ...reply })
      question.value = ''
    } else {
      error.value = 'supportChat.unavailable'
    }
  } catch {
    if (!request.signal.aborted) error.value = 'supportChat.error'
  } finally {
    pending.value = ''
    request = undefined
  }
}

function sendOnEnter(event: KeyboardEvent) {
  if (event.isComposing || event.keyCode === 229) return
  event.preventDefault()
  if (!event.repeat) void send()
}
</script>

<template>
  <div
    ref="transcript"
    class="support-transcript"
    :class="{ 'has-messages': turns.length || pending }"
    role="log"
    :aria-label="t('supportChat.title')"
    :tabindex="turns.length || pending ? 0 : undefined"
  >
    <article v-for="(turn, index) in turns" :key="index" class="support-turn">
      <p class="support-question" dir="auto">{{ turn.question }}</p>
      <p class="support-answer" dir="auto">{{ turn.answer }}</p>
      <div v-if="turn.sources.length" class="support-sources">
        <strong>{{ t('supportChat.sources') }}</strong>
        <ol>
          <li v-for="(source, sourceIndex) in turn.sources" :key="sourceIndex">
            <a v-if="source.url" :href="source.url" target="_blank" rel="noopener noreferrer">
              {{ source.title }}
            </a>
            <span v-else>{{ source.title }}</span>
            <span v-if="source.publisher"> · {{ source.publisher }}</span>
            <span v-if="source.page"> · {{ t('supportChat.page', { number: source.page }) }}</span>
          </li>
        </ol>
      </div>
    </article>
    <article v-if="pending" class="support-turn">
      <p class="support-question" dir="auto">{{ pending }}</p>
      <p class="support-answer support-pending">{{ t('supportChat.pending') }}</p>
    </article>
  </div>
  <form class="support-form" @submit.prevent="send">
    <label class="visually-hidden" for="support-question">{{ t('supportChat.question') }}</label>
    <div class="support-composer">
      <textarea
        id="support-question"
        v-model="question"
        rows="2"
        dir="auto"
        maxlength="2000"
        enterkeyhint="send"
        :placeholder="t('supportChat.question')"
        :readonly="Boolean(pending)"
        @keydown.enter.exact="sendOnEnter"
      />
      <button type="submit" :disabled="!question.trim() || Boolean(pending)">
        {{ t('supportChat.send') }}
      </button>
    </div>
    <p v-if="error" class="support-error" role="alert">{{ t(error) }}</p>
  </form>
</template>

<style scoped>
.support-transcript {
  display: grid;
  gap: 16px;
  min-height: 0;
  max-height: min(24rem, 50dvh);
  overflow-y: auto;
  overscroll-behavior: contain;
}

.support-transcript.has-messages {
  padding: 16px;
}

.support-turn {
  display: grid;
  min-width: 0;
  gap: 8px;
}

.support-question,
.support-answer {
  width: fit-content;
  max-width: 94%;
  margin: 0;
  padding: 9px 13px;
  border-radius: 16px;
  font-size: 0.9375rem;
  line-height: 1.55;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.support-question {
  justify-self: end;
  max-width: 85%;
  border-end-end-radius: 5px;
  background: var(--chat-accent);
  color: #fff;
}

.support-answer {
  border: 1px solid var(--chat-border);
  border-end-start-radius: 5px;
  background: #fff;
}

.support-pending {
  color: var(--chat-muted);
}

.support-sources {
  padding-inline: 4px;
  color: var(--chat-muted);
  font-size: 0.8125rem;
  overflow-wrap: anywhere;
}

.support-sources ol {
  margin: 4px 0 0;
  padding-inline-start: 1.4em;
}

.support-sources a {
  color: var(--chat-accent);
  text-decoration: underline;
  text-underline-offset: 2px;
}

.support-form {
  flex: none;
  padding: 12px 14px;
  background: #fff;
}

.has-messages + .support-form {
  border-top: 1px solid var(--chat-border);
}

.support-composer {
  display: flex;
  align-items: flex-end;
  gap: 8px;
  padding: 5px;
  border: 1px solid var(--chat-border);
  border-radius: 14px;
  background: var(--chat-background);
}

.support-composer:focus-within {
  border-color: var(--chat-accent);
}

.support-composer textarea {
  flex: 1;
  min-width: 0;
  padding: 8px;
  border: 0;
  background: transparent;
  color: inherit;
  font: inherit;
  font-size: 0.9375rem;
  resize: none;
}

.support-composer textarea:focus-visible {
  outline: 0;
}

.support-composer textarea::placeholder {
  color: var(--chat-muted);
}

.support-composer button {
  min-height: 44px;
  padding: 6px 14px;
  border: 0;
  border-radius: 10px;
  background: var(--chat-accent);
  color: #fff;
  font-size: 0.875rem;
  font-weight: 700;
}

.support-composer button:disabled {
  background: var(--chat-border);
  color: var(--chat-muted);
}

.support-error {
  margin: 8px 0 0;
  color: #943647;
  font-size: 0.875rem;
}
</style>
