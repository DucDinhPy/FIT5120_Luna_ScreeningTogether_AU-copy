import { mount } from '@vue/test-utils'
import { expect, it, vi } from 'vitest'
import { askSupport } from '@/api/supportChat'
import i18next from '@/i18n'
import SupportChatPanel from '../SupportChatPanel.vue'

vi.mock('@/api/supportChat', () => ({ askSupport: vi.fn() }))

it('sends with Enter and preserves Shift+Enter and input method composition', async () => {
  await i18next.changeLanguage('vi')
  const panel = mount(SupportChatPanel, { props: { open: true } })
  try {
    const input = panel.get('textarea')
    await input.setValue('what is cervical screening')
    const transcript = panel.get('[role="log"]')
    for (const modifiers of [{ shiftKey: true }, { isComposing: true }]) {
      const event = new KeyboardEvent('keydown', {
        key: 'Enter',
        cancelable: true,
        ...modifiers,
      })
      input.element.dispatchEvent(event)
      expect(event.defaultPrevented).toBe(false)
    }
    expect(transcript.text()).toBe('')

    await input.trigger('keydown', { key: 'Enter' })
    expect(transcript.text()).toContain(
      i18next.t('questions.what.answer', { ns: 'faq', lng: 'en' }),
    )
    expect(transcript.get('a').attributes('href')).toBe('/faq')
    expect(askSupport).not.toHaveBeenCalled()
  } finally {
    panel.unmount()
  }
})
