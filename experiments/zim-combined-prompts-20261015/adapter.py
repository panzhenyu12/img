"""Explicit mixed-prompt adapter for the pinned ZIM ONNX decoder.

Upstream handles box/point separately and overwrites point tokens when both
are passed. Keep the vendor immutable. Combined tokens are a new integration
path and require independent image-quality validation, not an official claim.
"""
import numpy as np
import torch


class CombinedPromptDecoder:
    def __init__(self, original, attention='point', bbox_attention=None):
        self.original = original
        self.ort_session = original.ort_session
        self.num_mask_tokens = original.num_mask_tokens
        self.attention = attention
        self.bbox_attention = bbox_attention
        self.last_prompt = None

    def __call__(self, interm_feats, image_embeddings, points, boxes, attn_mask):
        if points is None or boxes is None:
            self.last_prompt = {'combined': False}
            return self.original(interm_feats, image_embeddings, points, boxes, attn_mask)
        as_np = lambda t: t.detach().cpu().numpy()
        coords, labels = points
        corners = boxes.reshape(-1, 2, 2).float()
        if corners.shape[0] != coords.shape[0]:
            raise ValueError('Point and box prompt batch counts must match')
        # SAM-compatible explicit box corners. No no-box padding token is
        # inserted when box corners already terminate the prompt sequence.
        corner_labels = torch.tensor([2, 3], dtype=labels.dtype, device=labels.device).reshape(1, 2).expand(labels.shape[0], -1)
        all_coords = torch.cat([coords.float(), corners], dim=1)
        all_labels = torch.cat([labels.float(), corner_labels], dim=1)
        if self.attention == 'box':
            if self.bbox_attention is None:
                raise ValueError('Box attention requires the original model helper')
            attn_mask = self.bbox_attention(boxes)
        elif self.attention != 'point':
            raise ValueError('Unknown mixed-prompt attention policy')
        feed = {'feat_D0': as_np(interm_feats[0]), 'feat_D1': as_np(interm_feats[1]),
                'feat_D2': as_np(interm_feats[2]), 'image_embeddings': as_np(image_embeddings),
                'attn_mask': as_np(attn_mask), 'point_coords': as_np(all_coords),
                'point_labels': as_np(all_labels)}
        self.last_prompt = {'combined': True, 'attention': self.attention,
                            'point_coords': feed['point_coords'].tolist(),
                            'point_labels': feed['point_labels'].tolist(),
                            'attentionSum': float(feed['attn_mask'].sum())}
        masks, scores = self.ort_session.run(None, feed)
        return torch.from_numpy(masks).to(image_embeddings.device), torch.from_numpy(scores).to(image_embeddings.device)
