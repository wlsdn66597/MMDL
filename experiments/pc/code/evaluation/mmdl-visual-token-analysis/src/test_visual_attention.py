"""Numerical and coordinate checks, runnable with Python unittest."""
import unittest
from types import SimpleNamespace

import torch
from visual_attention import selected_attention, image_layout, anchors, AttentionTap


class AttentionTests(unittest.TestCase):
    def test_selected_rows_match_full_causal_grouped_attention(self):
        torch.manual_seed(42)
        for offset, count in [(0, 7), (5, 2)]:
            q = torch.randn(1, 4, count, 8)
            k = torch.randn(1, 2, offset + count, 8)
            full = q @ k.repeat_interleave(2, dim=1).transpose(-2, -1) / 8 ** .5
            causal = torch.arange(offset + count)[None, :] <= torch.arange(offset, offset + count)[:, None]
            full.masked_fill_(~causal, float('-inf'))
            expected = full.softmax(-1).mean(1)[0]
            for mask in [None, causal[None, None], torch.where(causal, 0., float('-inf'))[None, None]]:
                result = selected_attention(q, k, mask, [0, count - 1], offset, 2, 8 ** -.5)
                torch.testing.assert_close(result, expected[[0, count - 1]], atol=1e-6, rtol=1e-6)

    def test_multi_image_grid_and_contiguity(self):
        positions, shapes = image_layout([0, 9, 9, 1, 9, 9, 9, 9, 2], 9, [[1, 2, 4], [1, 4, 4]], 2)
        self.assertEqual(positions, [1, 2, 4, 5, 6, 7])
        self.assertEqual(shapes, [(1, 2), (2, 2)])
        with self.assertRaises(ValueError):
            image_layout([9, 0, 9], 9, [[1, 2, 4]], 2)
        with self.assertRaises(ValueError):
            image_layout([9, 9, 9], 9, [[1, 2, 4]], 2)

    def test_prediction_position_and_missing_final(self):
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: ''.join(map(chr, ids)))
        text = 'Reasoning. Final answer: B'
        result = anchors(tokenizer, 20, list(map(ord, text)))
        self.assertEqual(result['before_first_token'], 19)
        self.assertEqual(result['before_final_answer'], 20 + text.index('B') - 1)
        self.assertEqual(result['before_last_token'], 20 + len(text) - 2)
        self.assertNotIn('before_final_answer', anchors(tokenizer, 20, list(map(ord, 'unfinished'))))

    def test_tap_preserves_model_output_and_chunked_replay(self):
        from transformers import Qwen3VLTextConfig, Qwen3VLTextModel
        config = Qwen3VLTextConfig(vocab_size=64, hidden_size=32, intermediate_size=48,
                    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                    head_dim=8, rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2], 'mrope_interleaved': True})
        config._attn_implementation = 'sdpa'
        model = Qwen3VLTextModel(config).eval()
        ids = torch.arange(12)[None, :]
        with torch.no_grad():
            plain = model(ids, use_cache=False).last_hidden_state
            with AttentionTap([0, 3], {'first': 5, 'last': 11}, [1, 2, 3]) as full:
                tapped = model(ids, use_cache=False).last_hidden_state
            torch.testing.assert_close(plain, tapped, atol=0, rtol=0)
            with AttentionTap([0, 3], {'first': 5, 'last': 11}, [1, 2, 3]) as chunked:
                result = model(ids[:, :6], use_cache=True)
                chunked.offset = 6
                result = model(ids[:, 6:], past_key_values=result.past_key_values, use_cache=True,
                               cache_position=torch.arange(6, 12), attention_mask=torch.ones(1, 12))
            torch.testing.assert_close(plain[:, 6:], result.last_hidden_state, atol=1e-5, rtol=1e-5)
            for key in full.maps:
                torch.testing.assert_close(torch.from_numpy(full.maps[key]), torch.from_numpy(chunked.maps[key]), atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
