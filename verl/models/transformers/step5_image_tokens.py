"""Restore one raw placeholder per expanded Step5 image for rollout."""


def collapse_image_tokens(prompt_ids, token_ids, image_features, patch_features):
    output = []
    position = 0
    patch = token_ids['<im_patch>']
    image_start = token_ids['<im_start>']
    patch_start = token_ids['<patch_start>']

    def consume_block(start, opening, closing, count):
        end = start + count + 2
        expected = [opening] + [patch] * count + [closing]
        if prompt_ids[start:end] != expected:
            raise ValueError('Malformed expanded Step5 image token block')
        return end

    while position < len(prompt_ids):
        if prompt_ids[position] not in (patch_start, image_start):
            output.append(prompt_ids[position])
            position += 1
            continue
        while position < len(prompt_ids) and prompt_ids[position] == patch_start:
            position = consume_block(position, patch_start, token_ids['<patch_end>'], patch_features)
            if position < len(prompt_ids) and prompt_ids[position] == token_ids['<patch_newline>']:
                position += 1
        if position >= len(prompt_ids) or prompt_ids[position] != image_start:
            raise ValueError('Step5 patch blocks must end with a full image block')
        position = consume_block(position, image_start, token_ids['<im_end>'], image_features)
        output.append(patch)
    return output
