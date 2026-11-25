import { flushPromises, mount } from '@vue/test-utils'
import { expect, it, vi } from 'vitest'
import { createMemoryHistory, createRouter } from 'vue-router'
import App from '@/App.vue'
import { askSupport } from '@/api/supportChat'

vi.mock('@/api/supportChat', () => ({ askSupport: vi.fn() }))

it('keeps the conversation and switches between floating panels', async () => {
  const page = { template: '<main>Example page</main>' }
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [{ path: '/:pathMatch(.*)*', component: page }],
  })
  await router.push('/')
  vi.mocked(askSupport).mockResolvedValueOnce({ answer: 'Try GP Finder.', sources: [] })

  const wrapper = mount(App, { attachTo: document.body, global: { plugins: [router] } })
  try {
    const trigger = wrapper.get('[aria-controls="support-chat-popup"]')
    const popup = wrapper.get('#support-chat-popup')
    expect(popup.isVisible()).toBe(false)

    await trigger.trigger('click')
    await flushPromises()
    expect(trigger.attributes('aria-expanded')).toBe('true')
    expect(popup.isVisible()).toBe(true)
    expect(document.activeElement).toBe(popup.get('button').element)

    await popup.get('textarea').setValue('Where can I find a GP?')
    await popup.get('form').trigger('submit')
    await flushPromises()
    expect(popup.get('[role="log"]').text()).toContain('Try GP Finder.')

    await wrapper.get('button[aria-label="Open display options"]').trigger('click')
    expect(popup.isVisible()).toBe(false)
    expect(wrapper.find('aside.display-options:not(.support-chat-popup)').isVisible()).toBe(true)

    await router.push('/faq?supportChat=open')
    await flushPromises()
    expect(popup.isVisible()).toBe(true)
    expect(popup.text()).toContain('Try GP Finder.')
    expect(wrapper.find('aside.display-options:not(.support-chat-popup)').exists()).toBe(false)

    await popup.trigger('keydown', { key: 'Escape' })
    await flushPromises()
    expect(popup.isVisible()).toBe(false)
    expect(document.activeElement).toBe(trigger.element)
  } finally {
    wrapper.unmount()
  }
})
