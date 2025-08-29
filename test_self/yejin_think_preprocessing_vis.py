import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# 격자 크기
rows, cols = 4, 6

# 초기 블록 위치
yellow_block = {'row1': 1, 'col1': 1, 'row2': 2, 'col2': 1}
gray_blocks = [
    {'row1': 0, 'col1': 3, 'row2': 0, 'col2': 4},
    {'row1': 2, 'col1': 3, 'row2': 3, 'col2': 4}
]

# 노란 블록 이동 경로 예시
path_reloc = [(1,1), (1,2), (1,3), (1,4), (2,4), (3,4), (3,5)]

# 회색 블록 재배치 로그 예시 (원래 위치, 새 위치)
reloc_log = [((0, 3), (0, 0)), ((2, 3), (0, 5))]

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(-0.5, cols-0.5)
ax.set_ylim(-0.5, rows-0.5)
ax.set_xticks(range(cols))
ax.set_yticks(range(rows))
ax.grid(True)
ax.set_aspect('equal')
ax.set_xticklabels([])
ax.set_yticklabels([])

# 노란 블록 그리기
yellow_rect = patches.Rectangle(
    (yellow_block['col1']-0.5, rows-1-yellow_block['row2']-0.5),
    yellow_block['col2']-yellow_block['col1']+1,
    yellow_block['row2']-yellow_block['row1']+1,
    linewidth=2, edgecolor='gold', facecolor='yellow', alpha=0.9)
ax.add_patch(yellow_rect)

# 회색 블록 그리기
gray_patches = []
for g in gray_blocks:
    rect = patches.Rectangle(
        (g['col1']-0.5, rows-1-g['row2']-0.5),
        g['col2']-g['col1']+1, g['row2']-g['row1']+1,
        linewidth=1, edgecolor='black', facecolor='lightgray', alpha=0.8)
    gray_patches.append(rect)
    ax.add_patch(rect)

# 애니메이션 업데이트 함수
def update(frame):
    # 노란 블록 위치 이동
    pos = path_reloc[min(frame, len(path_reloc)-1)]
    yellow_rect.set_xy((pos[1]-0.5, rows-1-pos[0]-0.5))

    # 특정 프레임에서 회색 블록 재배치
    if frame == 2:
        gray_patches[0].set_xy((reloc_log[0][1][1]-0.5, rows-1-reloc_log[0][1][0]-0.5))
    if frame == 4:
        gray_patches[1].set_xy((reloc_log[1][1][1]-0.5, rows-1-reloc_log[1][1][0]-0.5))

    return [yellow_rect] + gray_patches

ani = animation.FuncAnimation(fig, update, frames=len(path_reloc), interval=1000, blit=True, repeat=False)

plt.title('Yellow Block Pathfinding with Gray Block Relocations')
plt.show()
