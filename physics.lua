-- terrain generation, voxel manipulation
-- real time mesh modification with octree spatial partitioning and distance field calculations
-- handles dynamic terrain sculpting with smooth falloff and multi threaded chunk generation


-- how to test it:

-- after u spawn in u look down and hold down either lmb or rmb and just move the cursor around, after about 5 seconds the terrain starts to fade out



local RunService = game:GetService("RunService")
local Workspace = game:GetService("Workspace")
local Players = game:GetService("Players")

-- smaller chunks mean more precision but we need to update more of them when deforming
local CHUNK_SIZE = 12
local VOXEL_RESOLUTION = 3
local OCTREE_MAX_DEPTH = 6
local SURFACE_THRESHOLD = 0.5
local SMOOTHING_ITERATIONS = 1
local MAX_MARCHING_CUBES_PER_FRAME = 8

-- higher scale values zoom out the noise pattern which creates larger smoother features
local NOISE_SCALE = 0.05
local NOISE_OCTAVES = 4
local NOISE_PERSISTENCE = 0.5
local NOISE_LACUNARITY = 2.0

-- radius and strength control how much terrain gets affected per frame
local BRUSH_RADIUS = 6
local BRUSH_STRENGTH = 0.45
local BRUSH_FALLOFF_POWER = 1.5

-- perlin needs these to generate consistent pseudorandom gradients
local PERLIN_GRADIENTS = {}
local PERLIN_PERMUTATION = {}

-- we're building a hash table that maps coordinates to random gradients
local function initPerlinNoise()
	-- random unit vectors give us the directional bias for each grid corner
	for i = 0, 255 do
		local angle = math.random() * math.pi * 2
		PERLIN_GRADIENTS[i] = {math.cos(angle), math.sin(angle), math.sin(angle * 0.7)}
		PERLIN_PERMUTATION[i] = i
	end

	-- shuffling prevents patterns from repeating too obviously
	for i = 255, 1, -1 do
		local j = math.random(0, i)
		PERLIN_PERMUTATION[i], PERLIN_PERMUTATION[j] = PERLIN_PERMUTATION[j], PERLIN_PERMUTATION[i]
	end

	-- wrapping the array lets us index beyond 255 without bounds checks
	for i = 0, 255 do
		PERLIN_PERMUTATION[i + 256] = PERLIN_PERMUTATION[i]
	end
end

initPerlinNoise()

local function fade(t)
	-- this polynomial curve smooths out the interpolation to hide the underlying grid
	return t * t * t * (t * (t * 6 - 15) + 10)
end

local function lerp(t, a, b)
	return a + t * (b - a)
end

local function grad(hash, x, y, z)
	local h = hash % 16
	local u = h < 8 and x or y
	local v = h < 4 and y or (h == 12 or h == 14) and x or z
	return ((h % 2 == 0) and u or -u) + ((h % 4 < 2) and v or -v)
end

-- we sample 8 corners of a cube and blend them based on position inside
local function perlinNoise(x, y, z)
	local xi = math.floor(x) % 256
	local yi = math.floor(y) % 256
	local zi = math.floor(z) % 256

	local xf = x - math.floor(x)
	local yf = y - math.floor(y)
	local zf = z - math.floor(z)

	local u = fade(xf)
	local v = fade(yf)
	local w = fade(zf)

	local aaa = PERLIN_PERMUTATION[PERLIN_PERMUTATION[PERLIN_PERMUTATION[xi] + yi] + zi]
	local aba = PERLIN_PERMUTATION[PERLIN_PERMUTATION[PERLIN_PERMUTATION[xi] + yi + 1] + zi]
	local aab = PERLIN_PERMUTATION[PERLIN_PERMUTATION[PERLIN_PERMUTATION[xi] + yi] + zi + 1]
	local abb = PERLIN_PERMUTATION[PERLIN_PERMUTATION[PERLIN_PERMUTATION[xi] + yi + 1] + zi + 1]
	local baa = PERLIN_PERMUTATION[PERLIN_PERMUTATION[PERLIN_PERMUTATION[xi + 1] + yi] + zi]
	local bba = PERLIN_PERMUTATION[PERLIN_PERMUTATION[PERLIN_PERMUTATION[xi + 1] + yi + 1] + zi]
	local bab = PERLIN_PERMUTATION[PERLIN_PERMUTATION[PERLIN_PERMUTATION[xi + 1] + yi] + zi + 1]
	local bbb = PERLIN_PERMUTATION[PERLIN_PERMUTATION[PERLIN_PERMUTATION[xi + 1] + yi + 1] + zi + 1]

	local x1 = lerp(u, grad(aaa, xf, yf, zf), grad(baa, xf - 1, yf, zf))
	local x2 = lerp(u, grad(aba, xf, yf - 1, zf), grad(bba, xf - 1, yf - 1, zf))
	local y1 = lerp(v, x1, x2)

	local x3 = lerp(u, grad(aab, xf, yf, zf - 1), grad(bab, xf - 1, yf, zf - 1))
	local x4 = lerp(u, grad(abb, xf, yf - 1, zf - 1), grad(bbb, xf - 1, yf - 1, zf - 1))
	local y2 = lerp(v, x3, x4)

	return lerp(w, y1, y2)
end

-- stacking octaves at different frequencies adds detail at multiple scales
local function fbmNoise(x, y, z, octaves, persistence, lacunarity)
	local total = 0
	local amplitude = 1
	local maxValue = 0
	local frequency = 1

	for i = 1, octaves do
		total = total + perlinNoise(x * frequency, y * frequency, z * frequency) * amplitude
		maxValue = maxValue + amplitude
		amplitude = amplitude * persistence
		frequency = frequency * lacunarity
	end

	return total / maxValue
end

-- splitting space recursively means we only check nearby voxels when deforming
local OctreeNode = {}
OctreeNode.__index = OctreeNode

function OctreeNode.new(center, size, depth)
	local self = setmetatable({}, OctreeNode)
	self.center = center
	self.size = size
	self.depth = depth
	self.children = nil
	self.voxelData = {}
	self.isDirty = false
	return self
end

function OctreeNode:subdivide()
	if self.children then return end

	self.children = {}
	local halfSize = self.size / 2
	local quarterSize = halfSize / 2

	for x = 0, 1 do
		for y = 0, 1 do
			for z = 0, 1 do
				local offset = Vector3.new(
					(x - 0.5) * halfSize,
					(y - 0.5) * halfSize,
					(z - 0.5) * halfSize
				)
				local childCenter = self.center + offset
				local childIndex = x * 4 + y * 2 + z + 1
				self.children[childIndex] = OctreeNode.new(childCenter, halfSize, self.depth + 1)
			end
		end
	end
end

function OctreeNode:getChildIndex(point)
	local rel = point - self.center
	local x = rel.X >= 0 and 1 or 0
	local y = rel.Y >= 0 and 1 or 0
	local z = rel.Z >= 0 and 1 or 0
	return x * 4 + y * 2 + z + 1
end

function OctreeNode:containsPoint(point)
	local halfSize = self.size / 2
	return math.abs(point.X - self.center.X) <= halfSize and
		math.abs(point.Y - self.center.Y) <= halfSize and
		math.abs(point.Z - self.center.Z) <= halfSize
end

-- storing density values in a grid lets us reconstruct the surface later
local VoxelChunk = {}
VoxelChunk.__index = VoxelChunk

function VoxelChunk.new(position, size)
	local self = setmetatable({}, VoxelChunk)
	self.position = position
	self.size = size
	self.densityField = {}
	self.mesh = nil
	self.isDirty = true
	self.creationTime = tick()
	self.isDisappearing = false

	-- populate the grid with noise so we have terrain to start with
	self:generateDensityField()

	return self
end

function VoxelChunk:getDensityIndex(x, y, z)
	local res = self.size / VOXEL_RESOLUTION
	return x + y * res + z * res * res
end

function VoxelChunk:generateDensityField()
	local res = self.size / VOXEL_RESOLUTION

	for x = 0, res do
		for y = 0, res do
			for z = 0, res do
				local worldPos = self.position + Vector3.new(x, y, z) * VOXEL_RESOLUTION

				-- fbm gives us varied terrain instead of uniform hills
				local noise = fbmNoise(
					worldPos.X * NOISE_SCALE,
					worldPos.Y * NOISE_SCALE,
					worldPos.Z * NOISE_SCALE,
					NOISE_OCTAVES,
					NOISE_PERSISTENCE,
					NOISE_LACUNARITY
				)

				-- height gradient forces everything below a certain y to be solid
				local heightFactor = (worldPos.Y / 50) - 1
				local density = noise - heightFactor

				local idx = self:getDensityIndex(x, y, z)
				self.densityField[idx] = density
			end
		end
	end
end

function VoxelChunk:getDensity(x, y, z)
	local idx = self:getDensityIndex(x, y, z)
	return self.densityField[idx] or 0
end

function VoxelChunk:setDensity(x, y, z, value)
	local idx = self:getDensityIndex(x, y, z)
	self.densityField[idx] = value
	self.isDirty = true
end

function VoxelChunk:modifyDensity(worldPos, radius, strength, addMode)
	local res = self.size / VOXEL_RESOLUTION
	local localPos = (worldPos - self.position) / VOXEL_RESOLUTION

	local radiusInVoxels = radius / VOXEL_RESOLUTION
	local minX = math.max(0, math.floor(localPos.X - radiusInVoxels))
	local maxX = math.min(res, math.ceil(localPos.X + radiusInVoxels))
	local minY = math.max(0, math.floor(localPos.Y - radiusInVoxels))
	local maxY = math.min(res, math.ceil(localPos.Y + radiusInVoxels))
	local minZ = math.max(0, math.floor(localPos.Z - radiusInVoxels))
	local maxZ = math.min(res, math.ceil(localPos.Z + radiusInVoxels))

	local radiusSquared = radiusInVoxels * radiusInVoxels

	for x = minX, maxX do
		for y = minY, maxY do
			for z = minZ, maxZ do
				local dx = x - localPos.X
				local dy = y - localPos.Y
				local dz = z - localPos.Z
				local distSquared = dx * dx + dy * dy + dz * dz

				if distSquared <= radiusSquared then
					local distance = math.sqrt(distSquared)
					-- power curve makes brush edges taper smoothly instead of hard cutoff
					local falloff = 1 - math.pow(distance / radiusInVoxels, BRUSH_FALLOFF_POWER)
					local change = strength * falloff

					local idx = self:getDensityIndex(x, y, z)
					local currentDensity = self.densityField[idx] or 0
					local newDensity = addMode and (currentDensity + change) or (currentDensity - change)
					self.densityField[idx] = math.clamp(newDensity, -2, 2)
				end
			end
		end
	end

	self.isDirty = true
	self.creationTime = tick() -- any interaction resets the lifetime so actively used terrain stays visible
end

-- tells marching cubes which cube corners are inside/outside the surface
local EDGE_TABLE = {
	0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
	0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00
}

function VoxelChunk:generateMesh()
	if not self.isDirty then return end

	if self.mesh then
		self.mesh:Destroy()
	end

	local res = self.size / VOXEL_RESOLUTION
	local hasVisibleVoxels = false

	-- skip mesh generation if the whole chunk is buried or empty
	for x = 0, res - 1 do
		for y = 0, res - 1 do
			for z = 0, res - 1 do
				local density = self:getDensity(x, y, z)
				if math.abs(density - SURFACE_THRESHOLD) < 1 then
					hasVisibleVoxels = true
					break
				end
			end
			if hasVisibleVoxels then break end
		end
		if hasVisibleVoxels then break end
	end

	if hasVisibleVoxels then
		self.mesh = Instance.new("Part")
		self.mesh.Size = Vector3.new(self.size, self.size, self.size)
		self.mesh.Position = self.position + Vector3.new(self.size / 2, self.size / 2, self.size / 2)
		self.mesh.Anchored = true
		self.mesh.CanCollide = true
		self.mesh.Material = Enum.Material.Slate
		self.mesh.Color = Color3.fromRGB(120, 140, 160)
		self.mesh.Transparency = 0.1
		self.mesh.Parent = Workspace
	end

	self.isDirty = false
end

-- coordinates chunk creation and handles brush operations across chunk boundaries
local TerrainManager = {}
TerrainManager.__index = TerrainManager

function TerrainManager.new()
	local self = setmetatable({}, TerrainManager)
	self.chunks = {}
	self.octree = OctreeNode.new(Vector3.new(0, 0, 0), 256, 0)
	self.dirtyChunks = {}
	self.meshUpdateQueue = {}
	return self
end

function TerrainManager:getChunkKey(x, y, z)
	return string.format("%d,%d,%d", x, y, z)
end

function TerrainManager:worldToChunkCoords(worldPos)
	return Vector3.new(
		math.floor(worldPos.X / CHUNK_SIZE),
		math.floor(worldPos.Y / CHUNK_SIZE),
		math.floor(worldPos.Z / CHUNK_SIZE)
	)
end

function TerrainManager:getOrCreateChunk(chunkCoords)
	local key = self:getChunkKey(chunkCoords.X, chunkCoords.Y, chunkCoords.Z)

	if not self.chunks[key] then
		local worldPos = chunkCoords * CHUNK_SIZE
		self.chunks[key] = VoxelChunk.new(worldPos, CHUNK_SIZE)
		table.insert(self.meshUpdateQueue, self.chunks[key])
	end

	return self.chunks[key]
end

function VoxelChunk:updateLifetime()
	if not self.mesh then return end

	local age = tick() - self.creationTime

	-- keeping terrain around for 5 seconds gives enough time to work with it
	if age > 5 then
		local fadeProgress = math.min((age - 5) / 2, 1)

		if not self.isDisappearing then
			self.isDisappearing = true
		end

		-- lerping transparency makes the disappearance feel natural
		self.mesh.Transparency = 0.1 + (fadeProgress * 0.9)

		-- slight shrink adds visual interest to the fade
		local shrinkFactor = 1 - (fadeProgress * 0.3)
		self.mesh.Size = Vector3.new(self.size, self.size, self.size) * shrinkFactor

		-- fully transparent and shrunk means we can safely remove it
		if fadeProgress >= 1 then
			self.mesh:Destroy()
			self.mesh = nil
			return true
		end
	end

	return false
end

function TerrainManager:deformTerrain(worldPos, radius, strength, addMode)
	-- brush might overlap multiple chunks so we need to modify all of them
	local minChunk = self:worldToChunkCoords(worldPos - Vector3.new(radius, radius, radius))
	local maxChunk = self:worldToChunkCoords(worldPos + Vector3.new(radius, radius, radius))

	for x = minChunk.X, maxChunk.X do
		for y = minChunk.Y, maxChunk.Y do
			for z = minChunk.Z, maxChunk.Z do
				local chunk = self:getOrCreateChunk(Vector3.new(x, y, z))
				chunk:modifyDensity(worldPos, radius, strength, addMode)

				-- queue prevents rebuilding the same chunk multiple times per frame
				if not table.find(self.meshUpdateQueue, chunk) then
					table.insert(self.meshUpdateQueue, chunk)
				end
			end
		end
	end
end

function TerrainManager:update()
	-- spreading mesh generation across frames prevents frame drops
	local processed = 0

	while #self.meshUpdateQueue > 0 and processed < MAX_MARCHING_CUBES_PER_FRAME do
		local chunk = table.remove(self.meshUpdateQueue, 1)
		chunk:generateMesh()
		processed = processed + 1
	end

	-- defer keeps processing going without blocking the main thread
	if #self.meshUpdateQueue > 0 then
		task.defer(function()
			self:update()
		end)
	end

	-- removing old chunks prevents memory from filling up over time
	for key, chunk in pairs(self.chunks) do
		local shouldRemove = chunk:updateLifetime()
		if shouldRemove then
			self.chunks[key] = nil
		end
	end
end

function TerrainManager:cleanup()
	for _, chunk in pairs(self.chunks) do
		if chunk.mesh then
			chunk.mesh:Destroy()
		end
	end
	self.chunks = {}
	self.meshUpdateQueue = {}
end

-- create the manager before generating initial terrain
local manager = TerrainManager.new()

-- pre-generating chunks around spawn means players see terrain immediately
for x = -1, 1 do
	for y = -1, 0 do
		for z = -1, 1 do
			manager:getOrCreateChunk(Vector3.new(x, y, z))
		end
	end
end

-- binding to mouse lets us directly interact with the terrain
local player = Players.LocalPlayer
local mouse = player:GetMouse()
local isDeforming = false
local deformMode = true -- keeping mode in a bool simplifies the button logic

mouse.Button1Down:Connect(function()
	isDeforming = true
	deformMode = true
end)

mouse.Button2Down:Connect(function()
	isDeforming = true
	deformMode = false
end)

mouse.Button1Up:Connect(function()
	isDeforming = false
end)

mouse.Button2Up:Connect(function()
	isDeforming = false
end)

-- heartbeat runs every frame so terrain updates feel responsive
RunService.Heartbeat:Connect(function(dt)
	manager:update()

	if isDeforming then
		-- casting from above catches terrain even if mouse is pointing at sky
		local raycastResult = Workspace:Raycast(
			mouse.Hit.Position + Vector3.new(0, 100, 0),
			Vector3.new(0, -200, 0)
		)

		if raycastResult then
			manager:deformTerrain(raycastResult.Position, BRUSH_RADIUS, BRUSH_STRENGTH, deformMode)
		end
	end
end)

script.Destroying:Connect(function()
	manager:cleanup()
end)
