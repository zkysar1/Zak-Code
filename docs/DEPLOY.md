# Keeping `zakcode webapp` running

The CLI cockpit relaunches itself on Enter because its view and its engine share one
process. In the webapp the engine **is** the server, so the one supervision job left is
keeping `zakcode webapp` alive on the box. A server launched by hand in a terminal is the
fragile spot: it dies with the terminal, with the SSH session, with the first crash, and
nobody notices until the mind stops answering. This page is the one supported answer —
one unit shape, one restart policy, no knobs.

## Linux box with systemd (the normal case)

The unit ships in the repo at [`deploy/systemd/zakcode-webapp.service`](../deploy/systemd/zakcode-webapp.service).
Its header repeats these steps. As root:

```bash
# 1. A service account and the workspace it serves (identity, rules, skills, sessions).
useradd --system --create-home --shell /usr/sbin/nologin zakcode
install -d -o zakcode -g zakcode /srv/zakcode/workspace

# 2. zakcode for that account — the command lands in ~zakcode/.local/bin.
sudo -u zakcode uv tool install "zakcode[server] @ git+https://github.com/zkysar1/Zak-Code.git"

# 3. Keys and settings, readable by the service account only.
install -o root -g zakcode -m 0640 /dev/null /etc/zakcode/.env
$EDITOR /etc/zakcode/.env      # ZAKCODE_DEFAULT_MODEL=..., the provider key, ZAKCODE_AUTH_TOKEN=... if exposed

# 4. The unit.
cp deploy/systemd/zakcode-webapp.service /etc/systemd/system/
systemd-analyze verify /etc/systemd/system/zakcode-webapp.service
systemctl daemon-reload && systemctl enable --now zakcode-webapp

# 5. It answers.
curl -s http://127.0.0.1:8000/health
```

What the unit fixes, and why:

| Line | Why |
| --- | --- |
| `Restart=always`, `RestartSec=3` | Every exit comes back — a crash, an OOM kill, a `zakcode update` that replaced the install — without a human relaunching it. |
| `StartLimitIntervalSec=0` | systemd's default gives up after 5 starts in 10 s. A bad key or a provider outage would otherwise leave the box without its mind until someone noticed. |
| `EnvironmentFile=-/etc/zakcode/.env` | The box-level config home. The `-` makes a missing file a no-op rather than a start failure; the per-user `~/.zakcode/.env` layering ([CONFIG.md](CONFIG.md)) still applies underneath. |
| `--host 127.0.0.1` | Loopback-only. To reach it from elsewhere put a reverse proxy in front, or set `ZAKCODE_AUTH_TOKEN` in the env file and bind `0.0.0.0` — the server refuses a non-loopback bind without a token unless you pass `--insecure`. |
| `--workspace /srv/zakcode/workspace` | The mind the server serves: one workspace per served mind, its sessions under `<workspace>/.zakcode/sessions/` (ADR-0032). |
| `NoNewPrivileges`, `PrivateTmp` | Hardening that stays out of the mind's way — it only needs its workspace and its config. |

Only the three lines marked `box-specific` in the unit change between boxes: the
workspace path, the `PATH` that holds the `zakcode` command for the service account, and
the bind. Everything else is the policy.

Day two:

```bash
journalctl -u zakcode-webapp -f                            # logs
sudo -u zakcode zakcode update && systemctl restart zakcode-webapp   # a new build
systemctl status zakcode-webapp                            # is it up, when did it last restart
```

`zakcode update` reinstalls in place from the install's own recorded source (a git URL or a
local checkout) and prints old → new build identity; the restart hands the new build to the
server. A `Restart=always` unit would also pick the new build up on the server's next exit,
but do not wait for one.

## Hand-managed hosts

A host nobody reaches by fleet automation gets the same unit, installed by hand with the
steps above — the deliverable there is the documented unit, not a script that logs in. Do
not run the server from a terminal "for now"; that is the exact failure this page exists
to end.

## Containers

The same policy, spelled in Compose. There is no image in this repo; build one from the
package (`pip install "zakcode[server]"`) and keep the workspace and the env file outside the
container so they outlive it:

```yaml
services:
  zakcode:
    image: your-registry/zakcode:latest
    command: ["zakcode", "webapp", "--host", "0.0.0.0", "--port", "8000", "--workspace", "/workspace"]
    env_file:
      - /etc/zakcode/.env            # must set ZAKCODE_AUTH_TOKEN: 0.0.0.0 inside the container is non-loopback
    volumes:
      - /srv/zakcode/workspace:/workspace
    ports:
      - "127.0.0.1:8000:8000"        # publish on loopback; a reverse proxy takes it from there
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://127.0.0.1:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
```

`restart: always` is the container form of the unit's policy; the health check gives the
orchestrator the same `/health` the unit's install step curls.

## Checking a box

```bash
systemctl is-enabled zakcode-webapp && systemctl is-active zakcode-webapp
systemd-analyze verify /etc/systemd/system/zakcode-webapp.service
curl -s http://127.0.0.1:8000/health
```

Three green answers mean the mind comes back on its own. Anything else is a hand-launched
server waiting to die.
