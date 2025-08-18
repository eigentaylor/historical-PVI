<script lang="ts">
  import { browser } from '$app/environment';
  import { page } from '$app/state';
  import { gotoLoadedMap, setLoadedMapFromJson } from '$lib/stores/LoadedMap';

  let status = $state('Waiting for data...');

  if (browser) {
    (async () => {
      try {
        const dataParam = page.url.searchParams.get('data');
        if (!dataParam) {
          status = 'No data parameter provided.';
          return;
        }
        // Base64 decode to string and parse
        const jsonStr = atob(dataParam);
        const json = JSON.parse(jsonStr);
        setLoadedMapFromJson(json);
        await gotoLoadedMap({ s: true });
        status = 'Redirecting...';
      } catch (err) {
        console.error(err);
        status = 'Failed to parse or load data.';
      }
    })();
  }
</script>

<svelte:head>
  <title>YAPms Embed Loader</title>
  <meta name="robots" content="noindex,nofollow" />
</svelte:head>

<div class="flex items-center justify-center h-full">
  <span>{status}</span>
  <noscript>Please enable JavaScript.</noscript>
  <div hidden aria-hidden="true">This page redirects after loading the map.</div>
</div>
