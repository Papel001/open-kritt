-- Add support for scan targets other than repositories.
-- Existing scans remain repository scans by default.

ALTER TABLE public.scans
    ADD COLUMN IF NOT EXISTS target_kind text NOT NULL DEFAULT 'repository',
    ADD COLUMN IF NOT EXISTS contract_address text,
    ADD COLUMN IF NOT EXISTS chain_id integer;

ALTER TABLE public.scans
    DROP CONSTRAINT IF EXISTS scans_target_kind_valid;

ALTER TABLE public.scans
    ADD CONSTRAINT scans_target_kind_valid
    CHECK (target_kind IN ('repository', 'contract'));

ALTER TABLE public.scans
    DROP CONSTRAINT IF EXISTS scans_contract_target_fields;

ALTER TABLE public.scans
    ADD CONSTRAINT scans_contract_target_fields
    CHECK (
        target_kind <> 'contract'
        OR (
            contract_address IS NOT NULL
            AND length(trim(contract_address)) > 0
            AND chain_id IS NOT NULL
            AND chain_id > 0
        )
    );
